#ifndef AOD_THREAD_SAFE_QUEUE_H
#define AOD_THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace aod {

// Thread-safe queue for passing commands between threads
template<typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() : shutdown_(false) {}

    // Push item to queue
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        cv_.notify_one();
    }

    void push(T&& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    // Pop item from queue (blocks until available or shutdown)
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);

        // Wait for item or shutdown
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });

        if (shutdown_ && queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Try to pop without blocking
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    // Check if queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // Get queue size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // Signal shutdown to wake waiting threads
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cv_.notify_all();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_;
};

} // namespace aod

#endif // AOD_THREAD_SAFE_QUEUE_H
