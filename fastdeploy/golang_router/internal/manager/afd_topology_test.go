package manager

import (
	"context"
	"reflect"
	"testing"

	"github.com/PaddlePaddle/FastDeploy/router/internal/config"
)

func TestAFDTopologyManagerBuildsReadyLayerLayout(t *testing.T) {
	m := NewAFDTopologyManager(&config.Config{
		Topology: config.TopologyConfig{EtcdPrefix: "/test/afd"},
	})

	ctx := context.Background()
	snapshot, err := m.Register(ctx, AFDExpertManifest{
		WorldSize:      4,
		AttnRanks:      []int{0, 1},
		FFNRanks:       []int{2, 3},
		GlobalRank:     2,
		InstanceURL:    "http://ffn-2",
		ExpertsPerRank: 2,
		Layers: []AFDLayerManifest{{
			LayerID: 0,
			Experts: []AFDExpertPlacement{
				{LogicalExpertID: 0, LocalExpertID: 0},
				{LogicalExpertID: 1, LocalExpertID: 1},
			},
		}},
	})
	if err != nil {
		t.Fatalf("register rank 2: %v", err)
	}
	if snapshot.State != "initializing" {
		t.Fatalf("first snapshot state = %s, want initializing", snapshot.State)
	}
	if !reflect.DeepEqual(snapshot.MissingFFNRanks, []int{3}) {
		t.Fatalf("missing ranks = %v, want [3]", snapshot.MissingFFNRanks)
	}

	snapshot, err = m.Register(ctx, AFDExpertManifest{
		WorldSize:      4,
		AttnRanks:      []int{0, 1},
		FFNRanks:       []int{2, 3},
		GlobalRank:     3,
		InstanceURL:    "http://ffn-3",
		ExpertsPerRank: 2,
		Layers: []AFDLayerManifest{{
			LayerID: 0,
			Experts: []AFDExpertPlacement{
				{LogicalExpertID: 2, LocalExpertID: 0},
				{LogicalExpertID: 3, LocalExpertID: 1},
			},
		}},
	})
	if err != nil {
		t.Fatalf("register rank 3: %v", err)
	}
	if snapshot.State != "ready" {
		t.Fatalf("ready snapshot state = %s, want ready", snapshot.State)
	}
	if len(snapshot.ExpertLayoutByLayer) != 1 {
		t.Fatalf("layer layout count = %d, want 1", len(snapshot.ExpertLayoutByLayer))
	}
	wantPhy2Log := []int{-1, -1, -1, -1, 0, 1, 2, 3}
	if !reflect.DeepEqual(snapshot.ExpertLayoutByLayer[0].Phy2Log, wantPhy2Log) {
		t.Fatalf("phy2log = %v, want %v", snapshot.ExpertLayoutByLayer[0].Phy2Log, wantPhy2Log)
	}
}

func TestAFDTopologyManagerRemoveInstanceDegradesReadyTopology(t *testing.T) {
	m := NewAFDTopologyManager(&config.Config{
		Topology: config.TopologyConfig{EtcdPrefix: "/test/afd"},
	})
	ctx := context.Background()

	manifests := []AFDExpertManifest{
		{
			WorldSize:      4,
			AttnRanks:      []int{0, 1},
			FFNRanks:       []int{2, 3},
			GlobalRank:     2,
			InstanceURL:    "http://ffn-2",
			ExpertsPerRank: 2,
			Layers: []AFDLayerManifest{{
				LayerID: 0,
				Experts: []AFDExpertPlacement{
					{LogicalExpertID: 0, LocalExpertID: 0},
					{LogicalExpertID: 1, LocalExpertID: 1},
				},
			}},
		},
		{
			WorldSize:      4,
			AttnRanks:      []int{0, 1},
			FFNRanks:       []int{2, 3},
			GlobalRank:     3,
			InstanceURL:    "http://ffn-3",
			ExpertsPerRank: 2,
			Layers: []AFDLayerManifest{{
				LayerID: 0,
				Experts: []AFDExpertPlacement{
					{LogicalExpertID: 2, LocalExpertID: 0},
					{LogicalExpertID: 3, LocalExpertID: 1},
				},
			}},
		},
	}
	for _, manifest := range manifests {
		if _, err := m.Register(ctx, manifest); err != nil {
			t.Fatalf("register rank %d: %v", manifest.GlobalRank, err)
		}
	}

	m.RemoveInstance(ctx, "http://ffn-3")
	snapshot, ok := m.Current()
	if !ok {
		t.Fatal("current snapshot missing")
	}
	if snapshot.State != "degraded" {
		t.Fatalf("snapshot state after remove = %s, want degraded", snapshot.State)
	}
	if !reflect.DeepEqual(snapshot.MissingFFNRanks, []int{3}) {
		t.Fatalf("missing ranks after remove = %v, want [3]", snapshot.MissingFFNRanks)
	}
	wantPhy2Log := []int{-1, -1, -1, -1, 0, 1, -1, -1}
	if !reflect.DeepEqual(snapshot.ExpertLayoutByLayer[0].Phy2Log, wantPhy2Log) {
		t.Fatalf("phy2log after remove = %v, want %v", snapshot.ExpertLayoutByLayer[0].Phy2Log, wantPhy2Log)
	}
}
