package config

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Log       LogConfig       `yaml:"log"`
	Manager   ManagerConfig   `yaml:"manager"`
	Scheduler SchedulerConfig `yaml:"scheduler"`
	Topology  TopologyConfig  `yaml:"topology"`
}

type ServerConfig struct {
	Name      string `yaml:"name"`
	Port      string `yaml:"port"`
	Host      string `yaml:"host"`
	Mode      string `yaml:"mode"` // debug, release, test
	Splitwise bool   `yaml:"splitwise"`
	AFD       bool   `yaml:"afd"`
}

type ManagerConfig struct {
	RegisterPath            string  `yaml:"register-path"`
	HealthFailureThreshold  int     `yaml:"health-failure-threshold"`
	HealthSuccessThreshold  int     `yaml:"health-success-threshold"`
	HealthCheckTimeoutSecs  float64 `yaml:"health-check-timeout-secs"`
	HealthCheckIntervalSecs float64 `yaml:"health-check-interval-secs"`
	HealthCheckEndpoint     string  `yaml:"health-check-endpoint"`
}

type SchedulerConfig struct {
	Policy               string  `yaml:"policy"`
	PrefillPolicy        string  `yaml:"prefill-policy"`
	DecodePolicy         string  `yaml:"decode-policy"`
	EvictionIntervalSecs float64 `yaml:"eviction-interval-secs"`
	EvictionDurationMins float64 `yaml:"eviction-duration-mins"`
	CacheBlockSize       int     `yaml:"cache-block-size"`
	TokenizerURL         string  `yaml:"tokenizer-url"`
	TokenizerTimeoutSecs float64 `yaml:"tokenizer-timeout-secs"`
	BalanceAbsThreshold  float64 `yaml:"balance-abs-threshold"`
	BalanceRelThreshold  float64 `yaml:"balance-rel-threshold"`
	HitRatioWeight       float64 `yaml:"hit-ratio-weight"`
	LoadBalanceWeight    float64 `yaml:"load-balance-weight"`
	WaitingWeight        float64 `yaml:"waiting-weight"`
	StatsIntervalSecs    float64 `yaml:"stats-interval-secs"`
}

type TopologyConfig struct {
	EtcdEndpoints []string `yaml:"etcd-endpoints"`
	EtcdPrefix    string   `yaml:"etcd-prefix"`
}

type LogConfig struct {
	Level  string `yaml:"level"`  // debug, info, warn, error
	Output string `yaml:"output"` // stdout, file
}

func Load(configPath, listenPort string, isSplitwise bool, isEnableAFD bool, topologyEtcdEndpoints string, topologyEtcdPrefix string) (*Config, error) {
	var cfg Config
	if configPath != "" {
		data, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}

		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse config: %w", err)
		}
	}

	// Set default values
	if listenPort != "" {
		cfg.Server.Port = listenPort
	} else if cfg.Server.Port == "" {
		return nil, fmt.Errorf("failed to set router listen port")
	}
	if isSplitwise {
		cfg.Server.Splitwise = true
	}
	if isEnableAFD {
		cfg.Server.AFD = true
	}
	if cfg.Server.AFD && !cfg.Server.Splitwise {
		return nil, fmt.Errorf("AFD mode requires --splitwise to be enabled")
	}
	if topologyEtcdEndpoints != "" {
		cfg.Topology.EtcdEndpoints = splitNonEmpty(topologyEtcdEndpoints, ",")
	}
	if topologyEtcdPrefix != "" {
		cfg.Topology.EtcdPrefix = topologyEtcdPrefix
	}
	if cfg.Topology.EtcdPrefix == "" {
		cfg.Topology.EtcdPrefix = "/fastdeploy/afd"
	}
	if cfg.Server.Mode == "" {
		cfg.Server.Mode = "release"
	}
	if cfg.Log.Level == "" {
		cfg.Log.Level = "info"
	}
	if cfg.Manager.HealthCheckEndpoint == "" {
		cfg.Manager.HealthCheckEndpoint = "/health"
	}
	if cfg.Manager.HealthCheckTimeoutSecs == 0 {
		cfg.Manager.HealthCheckTimeoutSecs = 5
	}
	if cfg.Manager.HealthCheckIntervalSecs == 0 {
		cfg.Manager.HealthCheckIntervalSecs = 5
	}
	if cfg.Manager.HealthFailureThreshold == 0 {
		cfg.Manager.HealthFailureThreshold = 1
	}
	if cfg.Manager.HealthSuccessThreshold == 0 {
		cfg.Manager.HealthSuccessThreshold = 1
	}
	if cfg.Scheduler.EvictionIntervalSecs == 0 {
		cfg.Scheduler.EvictionIntervalSecs = 60
	}
	if cfg.Scheduler.EvictionDurationMins == 0 {
		cfg.Scheduler.EvictionDurationMins = 30
	}
	if cfg.Scheduler.CacheBlockSize == 0 {
		cfg.Scheduler.CacheBlockSize = 64
	}
	if cfg.Scheduler.TokenizerTimeoutSecs == 0 {
		cfg.Scheduler.TokenizerTimeoutSecs = 2
	}
	if cfg.Scheduler.HitRatioWeight == 0 {
		cfg.Scheduler.HitRatioWeight = 1
	}
	if cfg.Scheduler.LoadBalanceWeight == 0 {
		cfg.Scheduler.LoadBalanceWeight = 1
	}
	if cfg.Scheduler.BalanceAbsThreshold == 0 {
		cfg.Scheduler.BalanceAbsThreshold = 1
	}
	if cfg.Scheduler.BalanceRelThreshold == 0 {
		cfg.Scheduler.BalanceRelThreshold = 0.2
	}
	if cfg.Scheduler.WaitingWeight == 0 {
		cfg.Scheduler.WaitingWeight = 1
	}
	if cfg.Scheduler.Policy == "" {
		cfg.Scheduler.Policy = "request_num"
	}
	if cfg.Scheduler.PrefillPolicy == "" {
		cfg.Scheduler.PrefillPolicy = "process_tokens"
	}
	if cfg.Scheduler.DecodePolicy == "" {
		cfg.Scheduler.DecodePolicy = "request_num"
	}
	if cfg.Scheduler.StatsIntervalSecs == 0 {
		cfg.Scheduler.StatsIntervalSecs = 5
	}
	return &cfg, nil
}

func splitNonEmpty(value string, sep string) []string {
	parts := strings.Split(value, sep)
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			result = append(result, part)
		}
	}
	return result
}
