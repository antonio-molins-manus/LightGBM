/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/dataset.h>
#include <LightGBM/utils/common.h>

#include <set>
#include <string>
#include <vector>

namespace LightGBM {

Metadata::Metadata() {
  num_weights_ = 0;
  num_init_score_ = 0;
  num_data_ = 0;
  num_queries_ = 0;
  num_positions_ = 0;
  weight_load_from_file_ = false;
  position_load_from_file_ = false;
  query_load_from_file_ = false;
  init_score_load_from_file_ = false;
  #ifdef USE_CUDA
  cuda_metadata_ = nullptr;
  #endif  // USE_CUDA
}

void Metadata::Init(const char* data_filename) {
  data_filename_ = data_filename;
  // for lambdarank, it needs query data for partition data in distributed learning
  LoadQueryBoundaries();
  LoadWeights();
  LoadPositions();
  CalculateQueryWeights();
  LoadInitialScore(data_filename_);
}

Metadata::~Metadata() {
}

void Metadata::Init(data_size_t num_data, int weight_idx, int query_idx) {
  num_data_ = num_data;
  label_ = std::vector<label_t>(num_data_);
  if (weight_idx >= 0) {
    if (!weights_.empty()) {
      Log::Info("Using weights in data file, ignoring the additional weights file");
      weights_.clear();
    }
    weights_ = std::vector<label_t>(num_data_, 0.0f);
    num_weights_ = num_data_;
    weight_load_from_file_ = false;
  }
  if (query_idx >= 0) {
    if (!query_boundaries_.empty()) {
      Log::Info("Using query id in data file, ignoring the additional query file");
      query_boundaries_.clear();
    }
    if (!query_weights_.empty()) { query_weights_.clear(); }
    queries_ = std::vector<data_size_t>(num_data_, 0);
    query_load_from_file_ = false;
  }
}

void Metadata::InitByReference(data_size_t num_data, const Metadata* reference) {
  int has_weights = reference->num_weights_ > 0;
  int has_init_scores = reference->num_init_score_ > 0;
  int has_queries = reference->num_queries_ > 0;
  int nclasses = reference->num_init_score_classes();
  Init(num_data, has_weights, has_init_scores, has_queries, nclasses);
}

void Metadata::Init(data_size_t num_data, int32_t has_weights, int32_t has_init_scores, int32_t has_queries, int32_t nclasses) {
  num_data_ = num_data;
  label_ = std::vector<label_t>(num_data_);
  if (has_weights) {
    if (!weights_.empty()) {
      Log::Fatal("Calling Init() on Metadata weights that have already been initialized");
    }
    weights_.resize(num_data_, 0.0f);
    num_weights_ = num_data_;
    weight_load_from_file_ = false;
  }
  if (has_init_scores) {
    if (!init_score_.empty()) {
      Log::Fatal("Calling Init() on Metadata initial scores that have already been initialized");
    }
    num_init_score_ = static_cast<int64_t>(num_data) * nclasses;
    init_score_.resize(num_init_score_, 0);
  }
  if (has_queries) {
    if (!query_weights_.empty()) {
      Log::Fatal("Calling Init() on Metadata queries that have already been initialized");
    }
    queries_.resize(num_data_, 0);
    query_load_from_file_ = false;
  }
}

void Metadata::Init(const Metadata& fullset, const data_size_t* used_indices, data_size_t num_used_indices) {
  num_data_ = num_used_indices;

  label_ = std::vector<label_t>(num_used_indices);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_used_indices >= 1024)
  for (data_size_t i = 0; i < num_used_indices; ++i) {
    label_[i] = fullset.label_[used_indices[i]];
  }

  if (!fullset.weights_.empty()) {
    weights_ = std::vector<label_t>(num_used_indices);
    num_weights_ = num_used_indices;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      weights_[i] = fullset.weights_[used_indices[i]];
    }
  } else {
    num_weights_ = 0;
  }

  if (!fullset.weights2_.empty()) {
    weights2_ = std::vector<label_t>(num_used_indices);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_used_indices >= 1024)
    for (data_size_t i = 0; i < num_used_indices; ++i) {
      weights2_[i] = fullset.weights2_[used_indices[i]];
    }
  }

  if (!fullset.init_score_.empty()) {
    int num_class = static_cast<int>(fullset.num_init_score_ / fullset.num_data_);
    init_score_ = std::vector<double>(static_cast<size_t>(num_used_indices) * num_class);
    num_init_score_ = static_cast<int64_t>(num_used_indices) * num_class;
    #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
    for (int k = 0; k < num_class; ++k) {
      const size_t offset_dest = static_cast<size_t>(k) * num_data_;
      const size_t offset_src = static_cast<size_t>(k) * fullset.num_data_;
      for (data_size_t i = 0; i < num_used_indices; ++i) {
        init_score_[offset_dest + i] = fullset.init_score_[offset_src + used_indices[i]];
      }
    }
  } else {
    num_init_score_ = 0;
  }

  if (!fullset.query_boundaries_.empty()) {
    std::vector<data_size_t> used_query;
    data_size_t data_idx = 0;
    for (data_size_t qid = 0; qid < num_queries_ && data_idx < num_used_indices; ++qid) {
      data_size_t start = fullset.query_boundaries_[qid];
      data_size_t end = fullset.query_boundaries_[qid + 1];
      data_size_t len = end - start;
      if (used_indices[data_idx] > start) {
        continue;
      } else if (used_indices[data_idx] == start) {
        if (num_used_indices >= data_idx + len && used_indices[data_idx + len - 1] == end - 1) {
          used_query.push_back(qid);
          data_idx += len;
        } else {
          Log::Fatal("Data partition error, data didn't match queries");
        }
      } else {
        Log::Fatal("Data partition error, data didn't match queries");
      }
    }
    query_boundaries_ = std::vector<data_size_t>(used_query.size() + 1);
    num_queries_ = static_cast<data_size_t>(used_query.size());
    query_boundaries_[0] = 0;
    for (data_size_t i = 0; i < num_queries_; ++i) {
      data_size_t qid = used_query[i];
      data_size_t len = fullset.query_boundaries_[qid + 1] - fullset.query_boundaries_[qid];
      query_boundaries_[i + 1] = query_boundaries_[i] + len;
    }
  } else {
    num_queries_ = 0;
  }
}

void Metadata::PartitionLabel(const std::vector<data_size_t>& used_indices) {
  if (used_indices.empty()) {
    return;
  }
  auto old_label = label_;
  num_data_ = static_cast<data_size_t>(used_indices.size());
  label_ = std::vector<label_t>(num_data_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
  for (data_size_t i = 0; i < num_data_; ++i) {
    label_[i] = old_label[used_indices[i]];
  }
  old_label.clear();
}

void Metadata::CalculateQueryBoundaries() {
  if (!queries_.empty()) {
    // need convert query_id to boundaries
    std::vector<data_size_t> tmp_buffer;
    data_size_t last_qid = -1;
    data_size_t cur_cnt = 0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (last_qid != queries_[i]) {
        if (cur_cnt > 0) {
          tmp_buffer.push_back(cur_cnt);
        }
        cur_cnt = 0;
        last_qid = queries_[i];
      }
      ++cur_cnt;
    }
    tmp_buffer.push_back(cur_cnt);
    query_boundaries_ = std::vector<data_size_t>(tmp_buffer.size() + 1);
    num_queries_ = static_cast<data_size_t>(tmp_buffer.size());
    query_boundaries_[0] = 0;
    for (size_t i = 0; i < tmp_buffer.size(); ++i) {
      query_boundaries_[i + 1] = query_boundaries_[i] + tmp_buffer[i];
    }
    CalculateQueryWeights();
    queries_.clear();
  }
}

void Metadata::CheckOrPartition(data_size_t num_all_data, const std::vector<data_size_t>& used_data_indices) {
  if (used_data_indices.empty()) {
    if (num_all_data != num_data_) {
      Log::Fatal("Number of data points mismatch with metadata, expected %d but got %d", num_all_data, num_data_);
    }
    // check weights
    if (!weights_.empty() && num_weights_ != num_data_) {
      weights_.clear();
      num_weights_ = 0;
      Log::Fatal("Weights size doesn't match data size, weights are ignored");
    }

    // check positions
    if (!positions_.empty() && num_positions_ != num_data_) {
      positions_.clear();
      num_positions_ = 0;
      Log::Fatal("Positions size doesn't match data size, positions are ignored");
    }

    // check query boundries
    if (!query_boundaries_.empty() && query_boundaries_[num_queries_] != num_data_) {
      query_boundaries_.clear();
      num_queries_ = 0;
      Log::Fatal("Query size doesn't match data size, queries are ignored");
    }

    // contain initial score file
    if (!init_score_.empty() && (num_init_score_ != num_data_ && num_init_score_ != static_cast<int64_t>(num_data_) * num_init_score_classes())) {
      init_score_.clear();
      num_init_score_ = 0;
      Log::Fatal("Initial score size doesn't match data size, initial scores are ignored");
    }
  } else {
    if (num_all_data != static_cast<data_size_t>(used_data_indices.size())) {
      Log::Fatal("Used data indices size doesn't match with num_all_data, expected %d but got %d", num_all_data, used_data_indices.size());
    }
    if (num_data_ != num_all_data) {
      // need partition label
      if (num_data_ < num_all_data) {
        Log::Fatal("Number of data points exceeds the metadata size, expected %d but got %d", num_data_, num_all_data);
      }
      PartitionLabel(used_data_indices);
      // partition weights
      if (!weights_.empty()) {
        auto old_weights = weights_;
        weights_ = std::vector<label_t>(num_data_);
        num_weights_ = num_data_;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
        for (data_size_t i = 0; i < num_data_; ++i) {
          weights_[i] = old_weights[used_data_indices[i]];
        }
        old_weights.clear();
      }

      // partition positions
      if (!positions_.empty()) {
        auto old_positions = positions_;
        positions_ = std::vector<data_size_t>(num_data_);
        num_positions_ = num_data_;
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static, 512) if (num_data_ >= 1024)
        for (data_size_t i = 0; i < num_data_; ++i) {
          positions_[i] = old_positions[used_data_indices[i]];
        }
        old_positions.clear();
      }

      // partition query boundaries
      if (!query_boundaries_.empty()) {
        std::vector<data_size_t> used_query;
        data_size_t data_idx = 0;
        for (data_size_t qid = 0; qid < num_queries_ && data_idx < static_cast<data_size_t>(used_data_indices.size()); ++qid) {
          data_size_t start = query_boundaries_[qid];
          data_size_t end = query_boundaries_[qid + 1];
          data_size_t len = end - start;
          if (used_data_indices[data_idx] > start) {
            continue;
          } else if (used_data_indices[data_idx] == start) {
            if (static_cast<data_size_t>(used_data_indices.size()) >= data_idx + len && used_data_indices[data_idx + len - 1] == end - 1) {
              used_query.push_back(qid);
              data_idx += len;
            } else {
              Log::Fatal("Data partition error, data didn't match queries");
            }
          } else {
            Log::Fatal("Data partition error, data didn't match queries");
          }
        }
        auto old_query_boundaries = query_boundaries_;
        query_boundaries_ = std::vector<data_size_t>(used_query.size() + 1);
        num_queries_ = static_cast<data_size_t>(used_query.size());
        query_boundaries_[0] = 0;
        for (data_size_t i = 0; i < num_queries_; ++i) {
          data_size_t qid = used_query[i];
          data_size_t len = old_query_boundaries[qid + 1] - old_query_boundaries[qid];
          query_boundaries_[i + 1] = query_boundaries_[i] + len;
        }
        old_query_boundaries.clear();
      }
      // partition initial scores
      if (!init_score_.empty()) {
        int num_class = num_init_score_classes();
        auto old_scores = init_score_;
        init_score_ = std::vector<double>(static_cast<size_t>(num_data_) * num_class);
        num_init_score_ = static_cast<int64_t>(num_data_) * num_class;
        #pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
        for (int k = 0; k < num_class; ++k) {
          const size_t offset_dest = static_cast<size_t>(k) * num_data_;
          const size_t offset_src = static_cast<size_t>(k) * num_all_data;
          for (size_t i = 0; i < static_cast<size_t>(used_data_indices.size()); ++i) {
            init_score_[offset_dest + i] = old_scores[offset_src + used_data_indices[i]];
          }
        }
        old_scores.clear();
      }
    }
  }
  CalculateQueryWeights();
}

void Metadata::SetLabel(const label_t* label, data_size_t len) {
  if (num_data_ != len) {
    Log::Fatal("Length of label is not same with #data");
  }
  std::copy(label, label + len, label_.begin());
}

void Metadata::SetLabel(const ArrowChunkedArray& array) {
  if (num_data_ != array.length()) {
    Log::Fatal("Length of label is not same with #data");
  }
  std::copy(array.begin<label_t>(), array.end<label_t>(), label_.begin());
}

void Metadata::SetPosition(const data_size_t* position, data_size_t len) {
  if (num_data_ != len) {
    Log::Fatal("Length of position is not same with #data");
  }
  if (positions_.empty()) {
    positions_.resize(num_data_);
    num_positions_ = num_data_;
  }
  std::copy(position, position + len, positions_.begin());
}

void Metadata::SetWeights(const ArrowChunkedArray& array) {
  SetWeightsFromIterator(array.begin<label_t>(), array.end<label_t>());
}

void Metadata::SetWeights2(const ArrowChunkedArray& array) {
  SetWeights2FromIterator(array.begin<label_t>(), array.end<label_t>());
}

void Metadata::SetQuery(const ArrowChunkedArray& array) {
  SetQueriesFromIterator(array.begin<data_size_t>(), array.end<data_size_t>());
}

void Metadata::SetInitScore(const ArrowChunkedArray& array) {
  SetInitScoresFromIterator(array.begin<double>(), array.end<double>());
}

template <typename It>
void Metadata::SetWeightsFromIterator(It first, It last) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (first == last) {
    weights_.clear();
    num_weights_ = 0;
    return;
  }
  data_size_t len = static_cast<data_size_t>(std::distance(first, last));
  // copy
  if (num_data_ != len) {
    Log::Fatal("Length of weights is not same with #data");
  }
  if (!weights_.empty()) { weights_.clear(); }
  num_weights_ = num_data_;
  weights_ = std::vector<label_t>(num_weights_);
  std::copy(first, last, weights_.begin());
  LoadQueryWeights();
  weight_load_from_file_ = false;
  #ifdef USE_CUDA
  if (cuda_metadata_ != nullptr) {
    cuda_metadata_->SetWeights(weights_.data(), weights_.size());
  }
  #endif  // USE_CUDA
}

template <typename It>
void Metadata::SetWeights2FromIterator(It first, It last) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (first == last) {
    weights2_.clear();
    return;
  }
  data_size_t len = static_cast<data_size_t>(std::distance(first, last));
  // copy
  if (num_data_ != len) {
    Log::Fatal("Length of weights2 is not same with #data");
  }
  if (!weights2_.empty()) { weights2_.clear(); }
  weights2_ = std::vector<label_t>(num_data_);
  std::copy(first, last, weights2_.begin());
}

void Metadata::SetWeights(const label_t* weights, data_size_t len) {
  SetWeightsFromIterator(weights, weights + len);
}

void Metadata::SetWeights2(const label_t* weights, data_size_t len) {
  SetWeights2FromIterator(weights, weights + len);
}

template <typename It>
void Metadata::SetQueriesFromIterator(It first, It last) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (first == last) {
    queries_.clear();
    num_queries_ = 0;
    return;
  }
  data_size_t len = static_cast<data_size_t>(std::distance(first, last));
  // copy
  if (num_data_ != len) {
    Log::Fatal("Length of query is not same with #data");
  }
  if (!queries_.empty()) { queries_.clear(); }
  queries_ = std::vector<data_size_t>(num_data_);
  std::copy(first, last, queries_.begin());
  CalculateQueryBoundaries();
  query_load_from_file_ = false;
}

void Metadata::SetQuery(const data_size_t* query, data_size_t len) {
  SetQueriesFromIterator(query, query + len);
}

template <typename It>
void Metadata::SetInitScoresFromIterator(It first, It last) {
  std::lock_guard<std::mutex> lock(mutex_);
  // save to nullptr
  if (first == last) {
    init_score_.clear();
    num_init_score_ = 0;
    return;
  }
  data_size_t len = static_cast<data_size_t>(std::distance(first, last));
  if (len % num_data_ != 0) {
    Log::Fatal("Length of init score is not same with #data");
  }
  if (!init_score_.empty()) { init_score_.clear(); }
  num_init_score_ = len;
  init_score_ = std::vector<double>(num_init_score_);
  std::copy(first, last, init_score_.begin());
  init_score_load_from_file_ = false;
  #ifdef USE_CUDA
  if (cuda_metadata_ != nullptr) {
    cuda_metadata_->SetInitScore(init_score_.data(), init_score_.size());
  }
  #endif  // USE_CUDA
}

void Metadata::SetInitScore(const double* init_score, data_size_t len) {
  SetInitScoresFromIterator(init_score, init_score + len);
}

void Metadata::InsertAt(data_size_t start_index, data_size_t count, const float* labels,
                        const float* weights, const double* init_scores, const int32_t* queries) {
  if (count <= 0) {
    return;
  }
  InsertLabels(labels, start_index, count);
  if (weights != nullptr) {
    InsertWeights(weights, start_index, count);
  }
  if (init_scores != nullptr) {
    InsertInitScores(init_scores, start_index, count, count);
  }
  if (queries != nullptr) {
    InsertQueries(queries, start_index, count);
  }
}

void Metadata::InsertLabels(const label_t* labels, data_size_t start_index, data_size_t len) {
  if (static_cast<size_t>(start_index) + static_cast<size_t>(len) > static_cast<size_t>(num_data_)) {
    Log::Fatal("Length of labels is larger than the capacity of current metadata, start_index: %d, len: %d, capacity: %d", start_index, len, num_data_);
  }
  std::copy(labels, labels + len, label_.begin() + start_index);
}

void Metadata::InsertWeights(const label_t* weights, data_size_t start_index, data_size_t len) {
  if (weights_.empty()) {
    if (start_index > 0) {
      weights_ = std::vector<label_t>(num_data_, 0.0f);
      num_weights_ = num_data_;
    } else if (static_cast<size_t>(len) == static_cast<size_t>(num_data_)) {
      weights_ = std::vector<label_t>(weights, weights + len);
      num_weights_ = num_data_;
    } else {
      Log::Fatal("Length of weights is not same with #data");
    }
  } else {
    if (static_cast<size_t>(start_index) + static_cast<size_t>(len) > static_cast<size_t>(num_weights_)) {
      Log::Fatal("Length of weights is larger than the capacity of current metadata, start_index: %d, len: %d, capacity: %d", start_index, len, num_weights_);
    }
    std::copy(weights, weights + len, weights_.begin() + start_index);
  }
}

void Metadata::InsertInitScores(const double* init_scores, data_size_t start_index, data_size_t len, data_size_t source_size) {
  if (init_score_.empty()) {
    if (start_index > 0) {
      init_score_ = std::vector<double>(num_data_, 0.0);
      num_init_score_ = num_data_;
    } else if (static_cast<size_t>(len) == static_cast<size_t>(num_data_)) {
      init_score_ = std::vector<double>(init_scores, init_scores + len);
      num_init_score_ = num_data_;
    } else {
      Log::Fatal("Length of init score is not same with #data");
    }
  } else {
    if (static_cast<size_t>(start_index) + static_cast<size_t>(len) > static_cast<size_t>(num_init_score_)) {
      Log::Fatal("Length of init score is larger than the capacity of current metadata, start_index: %d, len: %d, capacity: %d", start_index, len, num_init_score_);
    }
    std::copy(init_scores, init_scores + len, init_score_.begin() + start_index);
  }
}

void Metadata::InsertQueries(const data_size_t* queries, data_size_t start_index, data_size_t len) {
  if (queries_.empty()) {
    if (start_index > 0) {
      queries_ = std::vector<data_size_t>(num_data_, 0);
    } else if (static_cast<size_t>(len) == static_cast<size_t>(num_data_)) {
      queries_ = std::vector<data_size_t>(queries, queries + len);
    } else {
      Log::Fatal("Length of queries is not same with #data");
    }
  } else {
    if (static_cast<size_t>(start_index) + static_cast<size_t>(len) > static_cast<size_t>(num_data_)) {
      Log::Fatal("Length of queries is larger than the capacity of current metadata, start_index: %d, len: %d, capacity: %d", start_index, len, num_data_);
    }
    std::copy(queries, queries + len, queries_.begin() + start_index);
  }
}

void Metadata::LoadWeights() {
  num_weights_ = 0;
  std::string weight_filename(data_filename_);
  // default weight file name
  weight_filename.append(".weight");
  TextReader<size_t> reader(weight_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading weights...");
  num_weights_ = static_cast<data_size_t>(reader.Lines().size());
  weights_ = std::vector<label_t>(num_weights_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_weights_; ++i) {
    double tmp_weight = 0.0f;
    Common::Atof(reader.Lines()[i].c_str(), &tmp_weight);
    weights_[i] = static_cast<label_t>(tmp_weight);
  }
  weight_load_from_file_ = true;
  LoadQueryWeights();
  #ifdef USE_CUDA
  if (cuda_metadata_ != nullptr) {
    cuda_metadata_->SetWeights(weights_.data(), weights_.size());
  }
  #endif  // USE_CUDA
}

void Metadata::LoadPositions() {
  num_positions_ = 0;
  std::string position_filename(data_filename_);
  // default position file name
  position_filename.append(".position");
  TextReader<size_t> reader(position_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading positions...");
  num_positions_ = static_cast<data_size_t>(reader.Lines().size());
  positions_ = std::vector<data_size_t>(num_positions_);
  std::unordered_map<std::string, int> map;
  int max_position = 0;
  position_ids_.clear();
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_positions_; ++i) {
    int tmp_pos = 0;
    const char* str = reader.Lines()[i].c_str();
    Common::Atoi(str, &tmp_pos);
    positions_[i] = tmp_pos;
    #pragma omp critical
    {
      if (map.count(reader.Lines()[i]) == 0) {
        map[reader.Lines()[i]] = max_position;
        max_position++;
        position_ids_.push_back(reader.Lines()[i]);
      }
    }
  }
  position_load_from_file_ = true;
}

void Metadata::LoadQueryBoundaries() {
  num_queries_ = 0;
  std::string query_filename(data_filename_);
  // default query file name
  query_filename.append(".query");
  TextReader<size_t> reader(query_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading query boundaries...");
  query_boundaries_ = std::vector<data_size_t>(reader.Lines().size() + 1);
  num_queries_ = static_cast<data_size_t>(reader.Lines().size());
  query_boundaries_[0] = 0;
  for (size_t i = 0; i < reader.Lines().size(); ++i) {
    int tmp_cnt;
    const char* str = reader.Lines()[i].c_str();
    Common::Atoi(str, &tmp_cnt);
    query_boundaries_[i + 1] = query_boundaries_[i] + static_cast<data_size_t>(tmp_cnt);
  }
  query_load_from_file_ = true;
  CalculateQueryWeights();
}

void Metadata::LoadInitialScore(const std::string& data_filename) {
  num_init_score_ = 0;
  std::string init_score_filename(data_filename);
  init_score_filename.append(".init");
  TextReader<size_t> reader(init_score_filename.c_str(), false);
  reader.ReadAllLines();
  if (reader.Lines().empty()) {
    return;
  }
  Log::Info("Loading initial scores...");
  num_init_score_ = static_cast<data_size_t>(reader.Lines().size());
  init_score_ = std::vector<double>(num_init_score_);
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
  for (data_size_t i = 0; i < num_init_score_; ++i) {
    double tmp_score = 0.0f;
    Common::Atof(reader.Lines()[i].c_str(), &tmp_score);
    init_score_[i] = tmp_score;
  }
  init_score_load_from_file_ = true;
  #ifdef USE_CUDA
  if (cuda_metadata_ != nullptr) {
    cuda_metadata_->SetInitScore(init_score_.data(), init_score_.size());
  }
  #endif  // USE_CUDA
}

void Metadata::CalculateQueryWeights() {
  if (weights_.empty() || query_boundaries_.empty()) {
    return;
  }
  query_weights_.clear();
  Log::Info("Calculating query weights...");
  query_weights_ = std::vector<label_t>(num_queries_);
  for (data_size_t i = 0; i < num_queries_; ++i) {
    query_weights_[i] = 0.0f;
    for (data_size_t j = query_boundaries_[i]; j < query_boundaries_[i + 1]; ++j) {
      query_weights_[i] += weights_[j];
    }
    query_weights_[i] /= (query_boundaries_[i + 1] - query_boundaries_[i]);
  }
}

void Metadata::LoadFromMemory(const void* memory) {
  const char* mem_ptr = reinterpret_cast<const char*>(memory);

  num_data_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(data_size_t);
  num_weights_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(data_size_t);
  num_queries_ = *(reinterpret_cast<const data_size_t*>(mem_ptr));
  mem_ptr += sizeof(data_size_t);
  num_init_score_ = *(reinterpret_cast<const int64_t*>(mem_ptr));
  mem_ptr += sizeof(int64_t);

  if (num_data_ <= 0) {
    Log::Fatal("Invalid num_data_ value: %d", num_data_);
  }

  label_ = std::vector<label_t>(num_data_);
  std::memcpy(label_.data(), mem_ptr, sizeof(label_t) * num_data_);
  mem_ptr += sizeof(label_t) * num_data_;

  if (num_weights_ > 0) {
    weights_ = std::vector<label_t>(num_weights_);
    std::memcpy(weights_.data(), mem_ptr, sizeof(label_t) * num_weights_);
    mem_ptr += sizeof(label_t) * num_weights_;
    weight_load_from_file_ = true;
  }
  if (num_queries_ > 0) {
    query_boundaries_ = std::vector<data_size_t>(num_queries_ + 1);
    std::memcpy(query_boundaries_.data(), mem_ptr, sizeof(data_size_t) * (num_queries_ + 1));
    mem_ptr += sizeof(data_size_t) * (num_queries_ + 1);
    query_load_from_file_ = true;
  }
  if (num_init_score_ > 0) {
    init_score_ = std::vector<double>(num_init_score_);
    std::memcpy(init_score_.data(), mem_ptr, sizeof(double) * num_init_score_);
    mem_ptr += sizeof(double) * num_init_score_;
    init_score_load_from_file_ = true;
  }
  CalculateQueryWeights();
}

void Metadata::SaveBinaryToFile(BinaryWriter* writer) const {
  writer->AlignedWrite(&num_data_, sizeof(num_data_));
  writer->AlignedWrite(&num_weights_, sizeof(num_weights_));
  writer->AlignedWrite(&num_queries_, sizeof(num_queries_));
  writer->AlignedWrite(&num_init_score_, sizeof(num_init_score_));
  writer->AlignedWrite(label_.data(), sizeof(label_t) * num_data_);
  if (!weights_.empty()) {
    writer->AlignedWrite(weights_.data(), sizeof(label_t) * num_weights_);
  }
  if (!query_boundaries_.empty()) {
    writer->AlignedWrite(query_boundaries_.data(), sizeof(data_size_t) * (num_queries_ + 1));
  }
  if (!init_score_.empty()) {
    writer->AlignedWrite(init_score_.data(), sizeof(double) * num_init_score_);
  }
}

size_t Metadata::SizesInByte() const {
  size_t size = sizeof(num_data_) + sizeof(num_weights_)
    + sizeof(num_queries_) + sizeof(num_init_score_);
  size += sizeof(label_t) * num_data_;
  if (!weights_.empty()) {
    size += sizeof(label_t) * num_weights_;
  }
  if (!query_boundaries_.empty()) {
    size += sizeof(data_size_t) * (num_queries_ + 1);
  }
  if (!init_score_.empty()) {
    size += sizeof(double) * num_init_score_;
  }
  return size;
}

void Metadata::FinishLoad() {
  if (num_queries_ > 0) {
    // need convert query_id to boundaries
    std::vector<data_size_t> tmp_buffer;
    data_size_t last_qid = -1;
    data_size_t cur_cnt = 0;
    for (data_size_t i = 0; i < num_data_; ++i) {
      if (last_qid != queries_[i]) {
        if (cur_cnt > 0) {
          tmp_buffer.push_back(cur_cnt);
        }
        cur_cnt = 0;
        last_qid = queries_[i];
      }
      ++cur_cnt;
    }
    tmp_buffer.push_back(cur_cnt);
    query_boundaries_ = std::vector<data_size_t>(tmp_buffer.size() + 1);
    num_queries_ = static_cast<data_size_t>(tmp_buffer.size());
    query_boundaries_[0] = 0;
    for (size_t i = 0; i < tmp_buffer.size(); ++i) {
      query_boundaries_[i + 1] = query_boundaries_[i] + tmp_buffer[i];
    }
    CalculateQueryWeights();
    queries_.clear();
  }
  #ifdef USE_CUDA
  if (cuda_metadata_ != nullptr) {
    if (!weights_.empty()) {
      cuda_metadata_->SetWeights(weights_.data(), weights_.size());
    }
    if (!init_score_.empty()) {
      cuda_metadata_->SetInitScore(init_score_.data(), init_score_.size());
    }
  }
  #endif  // USE_CUDA
}

#ifdef USE_CUDA
void Metadata::CreateCUDAMetadata(const int gpu_device_id) {
  cuda_metadata_.reset(new CUDAMetadata(gpu_device_id, num_data_));
  if (!weights_.empty()) {
    cuda_metadata_->SetWeights(weights_.data(), weights_.size());
  }
  if (!init_score_.empty()) {
    cuda_metadata_->SetInitScore(init_score_.data(), init_score_.size());
  }
}
#endif  // USE_CUDA

}  // namespace LightGBM
