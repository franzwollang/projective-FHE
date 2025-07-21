#pragma once

#ifndef MOCK_OPENFHE
#include "openfhe.h"
#else
#include "mock_openfhe.h"
#endif

#include <vector>

using namespace lbcrypto;

class QCMDSEncoder {
private:
    CryptoContext<DCRTPoly> context;
    int k, p;
    std::vector<std::vector<double>> G;
    std::vector<std::vector<double>> G_pinv;
    
    // Cached plaintext coefficients for performance
    std::vector<std::vector<Plaintext>> G_plaintexts;  // G[i][j] as Plaintext objects
    std::vector<std::vector<Plaintext>> G_pinv_plaintexts;  // G_pinv[i][j] as Plaintext objects

public:
    QCMDSEncoder(CryptoContext<DCRTPoly> ctx, int p_val, int k_val);
    
    // Encode k logical ciphertexts into p redundant rows using QC-MDS
    std::vector<Ciphertext<DCRTPoly>> encode(const std::vector<Ciphertext<DCRTPoly>>& logical_cts);
    
    // Decode p redundant ciphertexts back to k logical ones using pseudoinverse
    std::vector<Ciphertext<DCRTPoly>> decode(const std::vector<Ciphertext<DCRTPoly>>& encoded_cts);
}; 