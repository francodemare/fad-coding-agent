package provider

import (
	"os"
	"strings"
)

// OllamaProvider wraps OpenAIProvider pointed at a local Ollama server's
// OpenAI-compatible /v1/ endpoint (see chapter 03). Wrapping in a distinct
// type lets ProviderKind distinguish "ollama" from "openai" and reports $0
// estimated cost for local inference.
type OllamaProvider struct {
	*OpenAIProvider
}

// NewOllamaProvider constructs a provider talking to Ollama's OpenAI-compatible
// endpoint. Honors $OLLAMA_BASE_URL or $OLLAMA_HOST if set, falling back to
// http://localhost:11434/v1/.
func NewOllamaProvider(model, system string, maxTokens int64) *OllamaProvider {
	return &OllamaProvider{
		OpenAIProvider: newOpenAIProviderWithBaseURL("ollama", model, system, maxTokens, ollamaBaseURL()),
	}
}

// EstimatedCostUSD returns 0 because local Ollama inference has no per-token
// API charge.
func (p *OllamaProvider) EstimatedCostUSD() float64 {
	return 0
}

func ollamaBaseURL() string {
	if u := strings.TrimSpace(os.Getenv("OLLAMA_BASE_URL")); u != "" {
		return u
	}
	if host := strings.TrimSpace(os.Getenv("OLLAMA_HOST")); host != "" {
		if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
			host = "http://" + host
		}
		host = strings.TrimRight(host, "/")
		if !strings.HasSuffix(host, "/v1") {
			host += "/v1"
		}
		return host + "/"
	}
	return "http://localhost:11434/v1/"
}
