//
// Created by pbustos on 22/02/25.
//

#ifndef PROTECTED_MAP_H
#define PROTECTED_MAP_H

#include <QtCore>
#include <map>
#include <shared_mutex>
#include <memory>
#include <optional>
#include <iostream>
#include <mutex> // Included <mutex>
#include "room.h"
#include "fridge.h"

/**
 * @brief A thread-safe map implementation using fine-grained locking.
 *
 * This class provides a map-like data structure that is safe for concurrent
 * access from multiple threads.  It uses a combination of a shared mutex to
 * protect the overall map structure and individual shared mutexes for each
 * key-value pair to allow for higher concurrency.
 *
 * @tparam Key The type of the keys in the map.
 * @tparam Value The type of the values in the map.
 */

namespace rc
{
    enum class ConceptsEnum {ROOM, FRIDGE};

    class ActionablesData
    {
        public:
        //private:
            std::shared_ptr<rc::Room> room;
            std::shared_ptr<rc::Fridge> fridge;
    };

    // class ActionablesData : public QObject {
    //     Q_OBJECT
    //
    // public:
    //     explicit ActionablesData(QObject* parent = nullptr) : QObject(parent) {}
    //
    //     // --- Data Access (Thread-Safe with Qt's mechanisms) ---
    //
    //     Q_INVOKABLE void setRoom(const QSharedPointer<rc::Room>& new_room) {
    //         // No need for explicit mutexes when using Qt's meta-object system correctly
    //         QSharedPointer<rc::Room> old_room;
    //         {
    //             QMutexLocker locker(&mutex_); //RAII style lock
    //             old_room = room;
    //             room = new_room;
    //         }
    //         emit roomChanged(old_room, new_room);
    //     }
    //
    //     Q_INVOKABLE QSharedPointer<rc::Room> getRoom() const {
    //         QMutexLocker locker(&mutex_);
    //         return room;  // Returns a copy; consider returning const ref if appropriate.
    //     }
    //
    //     Q_INVOKABLE void setFridge(const QSharedPointer<rc::Fridge>& new_fridge) {
    //         QSharedPointer<rc::Fridge> old_fridge;
    //         {
    //             QMutexLocker locker(&mutex_); //RAII style lock
    //             old_fridge = fridge;
    //             fridge = new_fridge;
    //         }
    //         emit fridgeChanged(old_fridge, new_fridge);
    //     }
    //
    //     Q_INVOKABLE QSharedPointer<rc::Fridge> getFridge() const {
    //         QMutexLocker locker(&mutex_);
    //         return fridge;
    //     }
    //
    //     signals:
    //         // Signals (automatically thread-safe when connected across threads)
    //         void roomChanged(const QSharedPointer<rc::Room>& oldRoom, const QSharedPointer<rc::Room>& newRoom);
    //     void fridgeChanged(const QSharedPointer<rc::Fridge>& oldFridge, const QSharedPointer<rc::Fridge>& newFridge);
    //
    // private:
    //     // --- Data Members ---
    //     QSharedPointer<rc::Room> room;
    //     QSharedPointer<rc::Fridge> fridge;
    //     mutable QMutex mutex_;
    // };


    // template <typename Key, typename Value>
    // class ProtectedMap
    // {
    // private:
    //     /**
    //      * @brief Inner structure to hold the value and its associated mutex.
    //      */
    //     struct Node
    //     {
    //         Value value;
    //         mutable std::shared_mutex mutex; //!< Mutex for this specific node.
    //
    //         /**
    //          * @brief Constructor for the Node.
    //          * @param val The initial value for the node.
    //          */
    //         Node(const Value& val) : value(val) {}
    //     };
    //
    //     std::map<Key, std::shared_ptr<Node>> m_map; //!< The underlying map.
    //     mutable std::shared_mutex m_map_mutex;      //!< Mutex to protect the map structure.
    //
    // public:
    //     /**
    //      * @brief Inserts or updates a key-value pair in the map.
    //      *
    //      * If the key already exists, its associated value is updated.  If the
    //      * key does not exist, a new key-value pair is inserted.
    //      *
    //      * @param key The key to insert or update.
    //      * @param value The value to associate with the key.
    //      */
    //     void set(const Key& key, const Value& value)
    //     {
    //         std::unique_lock<std::shared_mutex> map_lock(m_map_mutex);
    //
    //         auto it = m_map.find(key);
    //         if (it != m_map.end()) {
    //             std::unique_lock<std::shared_mutex> node_lock(it->second->mutex);
    //             it->second->value = value;
    //         } else {
    //             auto newNode = std::make_shared<Node>(value);
    //             m_map[key] = newNode;
    //         }
    //     }
    //
    //     /**
    //      * @brief Retrieves the value associated with a key.
    //      *
    //      * @param key The key to look up.
    //      * @return An optional containing the value if the key is found, or
    //      *         std::nullopt if the key is not found.
    //      */
    //     std::optional<Value> get(const Key& key) const
    //     {
    //         std::shared_lock<std::shared_mutex> map_lock(m_map_mutex);
    //
    //         auto it = m_map.find(key);
    //         if (it == m_map.end())
    //             return std::nullopt;
    //
    //         std::shared_lock<std::shared_mutex> node_lock(it->second->mutex);
    //         return it->second->value;
    //     }
    //
    //     /**
    //      * @brief Removes a key-value pair from the map.
    //      *
    //      * If the key is not found, this method does nothing.
    //      *
    //      * @param key The key to remove.
    //      */
    //     void erase(const Key& key)
    //     {
    //         std::unique_lock<std::shared_mutex> map_lock(m_map_mutex);
    //
    //         auto it = m_map.find(key);
    //         if (it != m_map.end())
    //         {
    //             std::unique_lock<std::shared_mutex> node_lock(it->second->mutex);
    //             m_map.erase(it);
    //         }
    //     }
    //
    //     /**
    //      * @brief Returns the number of key-value pairs in the map.
    //      *
    //      * @return The number of elements in the map.
    //      */
    //     size_t size() const
    //     {
    //         std::shared_lock<std::shared_mutex> lock(m_map_mutex);
    //         return m_map.size();
    //     }
    //
    //     class const_iterator
    //     {
    //         typename std::map<Key, std::shared_ptr<Node>>::const_iterator it;
    //
    //         public:
    //             const_iterator(typename std::map<Key, std::shared_ptr<Node>>::const_iterator map_it) : it(map_it) {}
    //
    //             const std::pair<const Key, Value> operator*() const
    //             {
    //                 std::shared_lock<std::shared_mutex> node_lock(it->second->mutex);
    //                 return {it->first, it->second->value};
    //             }
    //
    //             const_iterator& operator++()
    //             {
    //                 ++it;
    //                 return *this;
    //             }
    //
    //             bool operator!=(const const_iterator& other) const
    //             {
    //                 return it != other.it;
    //             }
    //     };
    //
    //     const_iterator begin() const
    //     {
    //         std::shared_lock<std::shared_mutex> lock(m_map_mutex);
    //         return const_iterator(m_map.cbegin());
    //     }
    //
    //     const_iterator end() const
    //     {
    //         std::shared_lock<std::shared_mutex> lock(m_map_mutex);
    //         return const_iterator(m_map.cend());
    //     }
    // };  // class ProtectedMap
};
#endif //PROTECTED_MAP_H
