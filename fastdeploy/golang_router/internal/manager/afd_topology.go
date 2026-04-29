package manager

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
	"github.com/PaddlePaddle/FastDeploy/router/pkg/logger"
	"github.com/gin-gonic/gin"
)

type AFDExpertManifest struct {
	WorldSize      int                `json:"world_size"`
	AttnRanks      []int              `json:"attn_ranks"`
	FFNRanks       []int              `json:"ffn_ranks"`
	GlobalRank     int                `json:"global_rank"`
	InstanceURL    string             `json:"instance_url"`
	ExpertsPerRank int                `json:"experts_per_rank"`
	Layers         []AFDLayerManifest `json:"layers"`
}

type AFDLayerManifest struct {
	LayerID int                  `json:"layer_id"`
	Experts []AFDExpertPlacement `json:"experts"`
}

type AFDExpertPlacement struct {
	LogicalExpertID int `json:"logical_expert_id"`
	LocalExpertID   int `json:"local_expert_id"`
}

type AFDTopologySnapshot struct {
	State               string           `json:"state"`
	Revision            int64            `json:"revision"`
	WorldTopology       AFDWorldSnapshot `json:"world_topology"`
	MissingFFNRanks     []int            `json:"missing_ffn_ranks"`
	ExpertLayoutByLayer []AFDLayerLayout `json:"expert_layout_by_layer"`
}

type AFDWorldSnapshot struct {
	WorldSize int   `json:"world_size"`
	AttnRanks []int `json:"attn_ranks"`
	FFNRanks  []int `json:"ffn_ranks"`
}

type AFDLayerLayout struct {
	LayerID int   `json:"layer_id"`
	Phy2Log []int `json:"phy2log"`
}

type AFDTopologyManager struct {
	mu               sync.RWMutex
	manifests        map[int]AFDExpertManifest
	instanceRanks    map[string]map[int]bool
	expectedFFNRanks []int
	worldSize        int
	attnRanks        []int
	everReady        bool
	revision         int64
	current          *AFDTopologySnapshot
	store            TopologyStore
	storePrefix      string
	watchers         map[int]chan AFDTopologySnapshot
	nextWatcherID    int
}

func NewAFDTopologyManager(cfg *config.Config) *AFDTopologyManager {
	store := TopologyStore(NoopTopologyStore{})
	if len(cfg.Topology.EtcdEndpoints) > 0 {
		store = NewEtcdHTTPTopologyStore(cfg.Topology.EtcdEndpoints)
	}
	return &AFDTopologyManager{
		manifests:     make(map[int]AFDExpertManifest),
		instanceRanks: make(map[string]map[int]bool),
		store:         store,
		storePrefix:   strings.TrimRight(cfg.Topology.EtcdPrefix, "/"),
		watchers:      make(map[int]chan AFDTopologySnapshot),
	}
}

func RegisterAFDExpertManifest(c *gin.Context) {
	if DefaultManager == nil || DefaultManager.topology == nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": "AFD topology manager is not enabled"})
		return
	}
	var manifest AFDExpertManifest
	if err := c.ShouldBindJSON(&manifest); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": err.Error()})
		return
	}
	snapshot, err := DefaultManager.topology.Register(c.Request.Context(), manifest)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 200, "msg": "success", "revision": snapshot.Revision, "state": snapshot.State})
}

func GetAFDTopology(c *gin.Context) {
	if DefaultManager == nil || DefaultManager.topology == nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": "AFD topology manager is not enabled"})
		return
	}
	waitReady := c.Query("wait_ready") == "true"
	timeoutSecs := 30
	if raw := c.Query("timeout_secs"); raw != "" {
		if parsed, err := parsePositiveInt(raw); err == nil {
			timeoutSecs = parsed
		}
	}

	if !waitReady {
		if snapshot, ok := DefaultManager.topology.Current(); ok {
			c.JSON(http.StatusOK, snapshot)
			return
		}
		c.Status(http.StatusNoContent)
		return
	}

	deadline := time.After(time.Duration(timeoutSecs) * time.Second)
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		if snapshot, ok := DefaultManager.topology.Current(); ok && snapshot.State == "ready" {
			c.JSON(http.StatusOK, snapshot)
			return
		}
		select {
		case <-c.Request.Context().Done():
			return
		case <-deadline:
			c.Status(http.StatusNoContent)
			return
		case <-ticker.C:
		}
	}
}

func WatchAFDTopology(c *gin.Context) {
	if DefaultManager == nil || DefaultManager.topology == nil {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": "AFD topology manager is not enabled"})
		return
	}
	since := int64(0)
	if raw := c.Query("revision"); raw != "" {
		if parsed, err := parsePositiveInt(raw); err == nil {
			since = int64(parsed)
		}
	}
	updates, cancel := DefaultManager.topology.Subscribe(since)
	defer cancel()

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Stream(func(w io.Writer) bool {
		select {
		case <-c.Request.Context().Done():
			return false
		case snapshot := <-updates:
			payload, _ := json.Marshal(snapshot)
			_, _ = fmt.Fprintf(w, "event: topology\ndata: %s\n\n", payload)
			return true
		}
	})
}

func (m *AFDTopologyManager) Register(ctx context.Context, manifest AFDExpertManifest) (*AFDTopologySnapshot, error) {
	if err := validateManifest(manifest); err != nil {
		return nil, err
	}

	m.mu.Lock()
	if err := m.applyManifestLocked(manifest); err != nil {
		m.mu.Unlock()
		return nil, err
	}
	snapshot, err := m.rebuildLocked()
	if err != nil {
		m.mu.Unlock()
		return nil, err
	}
	m.mu.Unlock()

	m.persist(ctx, snapshot)
	m.notify(snapshot)
	return snapshot, nil
}

func (m *AFDTopologyManager) RemoveInstance(ctx context.Context, instanceURL string) {
	m.mu.Lock()
	ranks := m.instanceRanks[instanceURL]
	if len(ranks) == 0 {
		m.mu.Unlock()
		return
	}
	for rank := range ranks {
		delete(m.manifests, rank)
	}
	delete(m.instanceRanks, instanceURL)
	snapshot, err := m.rebuildLocked()
	if err != nil {
		logger.Error(ctx, "Failed to rebuild AFD topology after removing %s: %v", instanceURL, err)
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()
	m.persist(ctx, snapshot)
	m.notify(snapshot)
}

func (m *AFDTopologyManager) Current() (*AFDTopologySnapshot, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.current == nil {
		return nil, false
	}
	snapshot := *m.current
	return &snapshot, true
}

func (m *AFDTopologyManager) Subscribe(since int64) (<-chan AFDTopologySnapshot, func()) {
	ch := make(chan AFDTopologySnapshot, 1)
	m.mu.Lock()
	id := m.nextWatcherID
	m.nextWatcherID++
	m.watchers[id] = ch
	if m.current != nil && m.current.Revision > since {
		ch <- *m.current
	}
	m.mu.Unlock()
	cancel := func() {
		m.mu.Lock()
		if watcher, ok := m.watchers[id]; ok {
			delete(m.watchers, id)
			close(watcher)
		}
		m.mu.Unlock()
	}
	return ch, cancel
}

func (m *AFDTopologyManager) applyManifestLocked(manifest AFDExpertManifest) error {
	if len(m.expectedFFNRanks) == 0 {
		m.expectedFFNRanks = sortedUnique(manifest.FFNRanks)
		m.worldSize = manifest.WorldSize
		m.attnRanks = sortedUnique(manifest.AttnRanks)
	} else {
		if !sameIntSet(m.expectedFFNRanks, manifest.FFNRanks) {
			return fmt.Errorf("ffn_ranks mismatch: expected=%v got=%v", m.expectedFFNRanks, manifest.FFNRanks)
		}
		if !sameIntSet(m.attnRanks, manifest.AttnRanks) {
			return fmt.Errorf("attn_ranks mismatch: expected=%v got=%v", m.attnRanks, manifest.AttnRanks)
		}
		if manifest.WorldSize != m.worldSize {
			return fmt.Errorf("world_size mismatch: expected=%d got=%d", m.worldSize, manifest.WorldSize)
		}
	}

	if old, ok := m.manifests[manifest.GlobalRank]; ok {
		if ranks := m.instanceRanks[old.InstanceURL]; ranks != nil {
			delete(ranks, old.GlobalRank)
		}
	}
	m.manifests[manifest.GlobalRank] = manifest
	if manifest.InstanceURL != "" {
		ranks := m.instanceRanks[manifest.InstanceURL]
		if ranks == nil {
			ranks = make(map[int]bool)
			m.instanceRanks[manifest.InstanceURL] = ranks
		}
		ranks[manifest.GlobalRank] = true
	}
	return nil
}

func (m *AFDTopologyManager) rebuildLocked() (*AFDTopologySnapshot, error) {
	m.revision++
	snapshot := &AFDTopologySnapshot{
		Revision: m.revision,
		WorldTopology: AFDWorldSnapshot{
			WorldSize: m.worldSize,
			AttnRanks: append([]int(nil), m.attnRanks...),
			FFNRanks:  append([]int(nil), m.expectedFFNRanks...),
		},
	}

	expertsPerRank := 0
	for rank, manifest := range m.manifests {
		if expertsPerRank == 0 {
			expertsPerRank = manifest.ExpertsPerRank
		}
		if manifest.ExpertsPerRank != expertsPerRank {
			return nil, fmt.Errorf("experts_per_rank mismatch: expected=%d got=%d rank=%d", expertsPerRank, manifest.ExpertsPerRank, rank)
		}
	}
	snapshot.MissingFFNRanks = missingRanks(m.expectedFFNRanks, m.manifests)

	if len(snapshot.MissingFFNRanks) == 0 && len(m.expectedFFNRanks) > 0 {
		snapshot.State = "ready"
		m.everReady = true
	} else if m.everReady {
		snapshot.State = "degraded"
	} else {
		snapshot.State = "initializing"
	}

	if m.worldSize > 0 && expertsPerRank > 0 {
		numPhysicalExperts := m.worldSize * expertsPerRank
		snapshot.ExpertLayoutByLayer = buildLayerLayouts(m.manifests, numPhysicalExperts, expertsPerRank)
	}
	m.current = snapshot
	return snapshot, nil
}

func buildLayerLayouts(manifests map[int]AFDExpertManifest, numPhysicalExperts int, expertsPerRank int) []AFDLayerLayout {
	layerMap := make(map[int][]int)
	for rank, manifest := range manifests {
		for _, layer := range manifest.Layers {
			phy2log := layerMap[layer.LayerID]
			if phy2log == nil {
				phy2log = make([]int, numPhysicalExperts)
				for i := range phy2log {
					phy2log[i] = -1
				}
				layerMap[layer.LayerID] = phy2log
			}
			for _, expert := range layer.Experts {
				physicalID := rank*expertsPerRank + expert.LocalExpertID
				if physicalID >= 0 && physicalID < len(phy2log) {
					phy2log[physicalID] = expert.LogicalExpertID
				}
			}
		}
	}
	layerIDs := make([]int, 0, len(layerMap))
	for layerID := range layerMap {
		layerIDs = append(layerIDs, layerID)
	}
	sort.Ints(layerIDs)
	layouts := make([]AFDLayerLayout, 0, len(layerIDs))
	for _, layerID := range layerIDs {
		layouts = append(layouts, AFDLayerLayout{LayerID: layerID, Phy2Log: layerMap[layerID]})
	}
	return layouts
}

func (m *AFDTopologyManager) persist(ctx context.Context, snapshot *AFDTopologySnapshot) {
	data, err := json.Marshal(snapshot)
	if err != nil {
		logger.Error(ctx, "Failed to marshal AFD topology snapshot: %v", err)
		return
	}
	if err := m.store.Put(ctx, m.storePrefix+"/current", data); err != nil {
		logger.Error(ctx, "Failed to persist AFD topology snapshot: %v", err)
	}
}

func (m *AFDTopologyManager) notify(snapshot *AFDTopologySnapshot) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, watcher := range m.watchers {
		select {
		case watcher <- *snapshot:
		default:
			select {
			case <-watcher:
			default:
			}
			select {
			case watcher <- *snapshot:
			default:
			}
		}
	}
}

func validateManifest(manifest AFDExpertManifest) error {
	if manifest.WorldSize <= 0 {
		return fmt.Errorf("world_size must be positive")
	}
	if manifest.GlobalRank < 0 || manifest.GlobalRank >= manifest.WorldSize {
		return fmt.Errorf("global_rank out of range: %d", manifest.GlobalRank)
	}
	if manifest.ExpertsPerRank <= 0 {
		return fmt.Errorf("experts_per_rank must be positive")
	}
	if len(manifest.AttnRanks) == 0 {
		return fmt.Errorf("attn_ranks must not be empty")
	}
	if len(manifest.FFNRanks) == 0 {
		return fmt.Errorf("ffn_ranks must not be empty")
	}
	if !containsInt(manifest.FFNRanks, manifest.GlobalRank) {
		return fmt.Errorf("global_rank %d is not in ffn_ranks %v", manifest.GlobalRank, manifest.FFNRanks)
	}
	seenRanks := make(map[int]string, manifest.WorldSize)
	for _, rank := range manifest.AttnRanks {
		if rank < 0 || rank >= manifest.WorldSize {
			return fmt.Errorf("attn rank out of range: %d", rank)
		}
		seenRanks[rank] = "attn"
	}
	for _, rank := range manifest.FFNRanks {
		if rank < 0 || rank >= manifest.WorldSize {
			return fmt.Errorf("ffn rank out of range: %d", rank)
		}
		if seenRanks[rank] != "" {
			return fmt.Errorf("rank %d appears in both attn_ranks and ffn_ranks", rank)
		}
		seenRanks[rank] = "ffn"
	}
	if len(seenRanks) != manifest.WorldSize {
		return fmt.Errorf("attn_ranks and ffn_ranks must cover world_size=%d, got %d ranks", manifest.WorldSize, len(seenRanks))
	}
	for _, layer := range manifest.Layers {
		if layer.LayerID < 0 {
			return fmt.Errorf("layer_id must be non-negative, got %d", layer.LayerID)
		}
		seenLocalExperts := make(map[int]bool)
		for _, expert := range layer.Experts {
			if expert.LogicalExpertID < 0 {
				return fmt.Errorf("logical_expert_id must be non-negative, got %d", expert.LogicalExpertID)
			}
			if expert.LocalExpertID < 0 || expert.LocalExpertID >= manifest.ExpertsPerRank {
				return fmt.Errorf("invalid local_expert_id=%d for rank=%d layer=%d", expert.LocalExpertID, manifest.GlobalRank, layer.LayerID)
			}
			if seenLocalExperts[expert.LocalExpertID] {
				return fmt.Errorf("duplicate local_expert_id=%d for rank=%d layer=%d", expert.LocalExpertID, manifest.GlobalRank, layer.LayerID)
			}
			seenLocalExperts[expert.LocalExpertID] = true
		}
	}
	return nil
}

func sortedUnique(values []int) []int {
	set := make(map[int]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	result := make([]int, 0, len(set))
	for value := range set {
		result = append(result, value)
	}
	sort.Ints(result)
	return result
}

func sameIntSet(expected []int, actual []int) bool {
	actualSorted := sortedUnique(actual)
	if len(expected) != len(actualSorted) {
		return false
	}
	for i := range expected {
		if expected[i] != actualSorted[i] {
			return false
		}
	}
	return true
}

func containsInt(values []int, needle int) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}

func missingRanks(expected []int, manifests map[int]AFDExpertManifest) []int {
	missing := make([]int, 0)
	for _, rank := range expected {
		if _, ok := manifests[rank]; !ok {
			missing = append(missing, rank)
		}
	}
	return missing
}

func parsePositiveInt(raw string) (int, error) {
	var value int
	_, err := fmt.Sscanf(raw, "%d", &value)
	if err != nil || value < 0 {
		return 0, fmt.Errorf("invalid positive integer: %s", raw)
	}
	return value, nil
}

type TopologyStore interface {
	Put(ctx context.Context, key string, value []byte) error
}

type NoopTopologyStore struct{}

func (NoopTopologyStore) Put(ctx context.Context, key string, value []byte) error {
	return nil
}

type EtcdHTTPTopologyStore struct {
	endpoints []string
	client    *http.Client
}

func NewEtcdHTTPTopologyStore(endpoints []string) *EtcdHTTPTopologyStore {
	cleaned := make([]string, 0, len(endpoints))
	for _, endpoint := range endpoints {
		endpoint = strings.TrimSpace(endpoint)
		if endpoint == "" {
			continue
		}
		if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
			endpoint = "http://" + endpoint
		}
		cleaned = append(cleaned, strings.TrimRight(endpoint, "/"))
	}
	return &EtcdHTTPTopologyStore{
		endpoints: cleaned,
		client:    &http.Client{Timeout: 3 * time.Second},
	}
}

func (s *EtcdHTTPTopologyStore) Put(ctx context.Context, key string, value []byte) error {
	if len(s.endpoints) == 0 {
		return nil
	}
	body, err := json.Marshal(map[string]string{
		"key":   base64.StdEncoding.EncodeToString([]byte(key)),
		"value": base64.StdEncoding.EncodeToString(value),
	})
	if err != nil {
		return err
	}
	var lastErr error
	for _, endpoint := range s.endpoints {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint+"/v3/kv/put", bytes.NewReader(body))
		if err != nil {
			lastErr = err
			continue
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := s.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		lastErr = fmt.Errorf("etcd endpoint %s returned status %d", endpoint, resp.StatusCode)
	}
	return lastErr
}
