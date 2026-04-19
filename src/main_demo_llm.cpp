//
// Created by 69029 on 4/12/2021.
//

#undef NDEBUG
#include "circuit.h"
#include "neuralNetwork.hpp"
#include "verifier.hpp"
#include "models.hpp"
#include "global_var.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <map>

#include "range_prover.hpp"
#include "hyrax_rp.hpp"
using namespace mcl::bn;
using namespace std;

struct ModelConfig {
    int n_layers;
    int n_heads;
    int head_dim;   // attn_dim / n_heads
    int attn_dim;
    int linear_dim;
    int seq_len;
};

// Configs mirroring PiFormer benchmark.py for fair comparison.
// head_dim = attn_dim / n_heads
static const map<string, ModelConfig> CONFIGS = {
    // name       layers  heads  head_dim  attn_dim  linear_dim  seq_len
    {"tiny",    {1,  1,  8,   8,   16,  8}},
    {"small",   {2,  1,  64,  64,  256, 16}},
    {"medium",  {4,  1,  128, 128, 512, 32}},
    {"large",   {6,  1,  256, 256, 1024,32}},
    // GPT-2 small: use power-of-2 approximation matching PiFormer's gpt2-small config
    {"gpt2-small", {12, 1, 512, 512, 2048, 32}},
    // Original GPT-2 shape (retained for backward compat)
    {"gpt2",    {12, 12, 64, 768, 2304, 30}},
};

static void print_usage(const char* prog) {
    cerr << "Usage: " << prog << " [--config NAME] [--layers N] [--heads N] "
         << "[--attn-dim N] [--linear-dim N] [--seq-len N] [--threads N]\n\n";
    cerr << "Named configs (matching PiFormer benchmark.py):\n";
    for (auto& kv : CONFIGS) {
        auto& c = kv.second;
        cerr << "  " << kv.first
             << ": layers=" << c.n_layers
             << " heads=" << c.n_heads
             << " attn_dim=" << c.attn_dim
             << " linear_dim=" << c.linear_dim
             << " seq_len=" << c.seq_len << "\n";
    }
    cerr << "\nDefaults to 'gpt2' if no config or flags are given.\n";
}

int main(int argc, char **argv)
{
    initPairing(mcl::BN254);

    // Defaults: original GPT-2 shape
    ModelConfig cfg = CONFIGS.at("gpt2");
    int threads = 32;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--config" && i + 1 < argc) {
            string name = argv[++i];
            if (CONFIGS.count(name) == 0) {
                cerr << "Unknown config '" << name << "'. Use --help to list options.\n";
                return 1;
            }
            cfg = CONFIGS.at(name);
        } else if (arg == "--layers" && i + 1 < argc) {
            cfg.n_layers = stoi(argv[++i]);
        } else if (arg == "--heads" && i + 1 < argc) {
            cfg.n_heads = stoi(argv[++i]);
            cfg.head_dim = cfg.attn_dim / cfg.n_heads;
        } else if (arg == "--attn-dim" && i + 1 < argc) {
            cfg.attn_dim = stoi(argv[++i]);
            cfg.head_dim = cfg.attn_dim / cfg.n_heads;
        } else if (arg == "--linear-dim" && i + 1 < argc) {
            cfg.linear_dim = stoi(argv[++i]);
        } else if (arg == "--seq-len" && i + 1 < argc) {
            cfg.seq_len = stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = stoi(argv[++i]);
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    cerr << "Running with: layers=" << cfg.n_layers
         << " heads=" << cfg.n_heads
         << " head_dim=" << cfg.head_dim
         << " attn_dim=" << cfg.attn_dim
         << " linear_dim=" << cfg.linear_dim
         << " seq_len=" << cfg.seq_len
         << " threads=" << threads << "\n";

    // range prover (heap-allocated: range_prover::g[4096] and neuralNetwork::table[655360] are too large for stack)
    auto rp = unique_ptr<range_prover>(new range_prover(
        cfg.n_layers, cfg.n_heads, cfg.head_dim,
        cfg.attn_dim, cfg.linear_dim, cfg.seq_len,
        threads, 1));
    rp->init();
    rp->build();
    double range_prover_time = rp->prove();

    // gkr (heap-allocated for same reason)
    auto p = unique_ptr<prover>(new prover());
    auto nn = unique_ptr<LLM>(new LLM(cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.attn_dim, cfg.linear_dim, cfg.seq_len));
    nn->create(*p, 1);
    verifier v(p.get(), p->C);
    v.range_prove(range_prover_time);
    v.prove(threads);
}
