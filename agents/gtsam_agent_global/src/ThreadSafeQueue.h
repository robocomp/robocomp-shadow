#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <mutex>
#include <vector>
#include <tuple>
#include <boost/circular_buffer.hpp>
#include <Eigen/Dense>

// Template genérico para cualquier tipo T
template <typename T>
class ThreadSafeCircularQueue {
private:
    boost::circular_buffer<T> buffer_;
    mutable std::mutex mtx_;

public:
    // Constructor: define la capacidad máxima
    explicit ThreadSafeCircularQueue(size_t capacity) : buffer_(capacity) {}

    // Meter datos (Productor) - Sobrescribe si está lleno
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_.push_back(item);
    }

    // Sacar todo (Consumidor) - Drenaje atómico
    std::vector<T> pop_all() {
        std::vector<T> data;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (buffer_.empty()) return data;

            data.reserve(buffer_.size());
            data.insert(data.end(), buffer_.begin(), buffer_.end());
            buffer_.clear();
        }
        return data;
    }

    // Verificar si está vacío
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return buffer_.empty();
    }

    // Limpiar manualmente (útil para resets)
    void clear() {
        std::lock_guard<std::mutex> lock(mtx_);
        buffer_.clear();
    }
};

#endif