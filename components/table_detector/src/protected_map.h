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
            std::shared_ptr<rc::Table> table;
    };

};
#endif //PROTECTED_MAP_H
