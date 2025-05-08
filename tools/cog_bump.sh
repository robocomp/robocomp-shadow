#!/bin/bash

# ===========================
# VERIFICACIÓN DE VARIABLES
# ===========================
if [ -z "$ROBOCOMP" ]; then

    echo "ROBOCOMP environment variable not set, using the default value /home/robocomp/robocomp"
    ROBOCOMP="/home/robocomp/robocomp"
fi

# ===========================
# CONFIGURACIÓN DE COMPONENTES
# ===========================
COMPILE="cmake -B build && make -C build -j$(( $(nproc) / 2 ))"

declare -A COMPONENTS=(
    ["base_controller_agent"]="cd $ROBOCOMP/components/robocomp-shadow/agents/base_controller_agent && $COMPILE && bin/base_controller_agent etc/config"
    ["inner_simulator_agent"]="cd $ROBOCOMP/components/robocomp-shadow/agents/inner_simulator_agent && $COMPILE && bin/inner_simulator_agent etc/config"
    ["mission_monitoring"]="cd $ROBOCOMP/components/robocomp-shadow/agents/mission_monitoring && $COMPILE && bin/mission_monitoring etc/config"
    ["semantic_memory"]="cd $ROBOCOMP/components/robocomp-shadow/agents/semantic_memory && $COMPILE && bin/semantic_memory etc/config"
    ["episodic_memory"]="cd $ROBOCOMP/components/robocomp-shadow/agents/episodic_memory && $COMPILE && bin/episodic_memory etc/config"
    ["person_detector"]="cd $ROBOCOMP/components/robocomp-shadow/agents/person_detector && $COMPILE && bin/person_detector etc/config"
)

function show_tabs() {
    echo "=== List of existing tabs ==="
    session_ids=($(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.sessionIdList | tr ',' ' '))
    if [ $? -ne 0 ]; then
        echo "Error: Failed to get session IDs"
        return 1
    fi

    for session_id in "${session_ids[@]}"; do
        tab_title=$(qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.tabTitle "$session_id")
        if [ $? -ne 0 ]; then
            echo "Error: Failed to get tab title for session ID $session_id"
            continue
        fi

        echo "Tab ID: $session_id, Title: $tab_title"
    done
}

# ===========================
# ELIMINAR PESTAÑAS Y PROCESOS EXISTENTES
# ===========================
function clean_tabs() {
    echo "=== Cleaning tabs ==="
    session_ids=($(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.sessionIdList | tr ',' ' '))
    if [ $? -ne 0 ]; then
        echo "Error: Failed to get session IDs"
        return 1
    fi

    for session_id in "${session_ids[@]}"; do
        tab_title=$(qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.tabTitle "$session_id")
        if [ $? -ne 0 ]; then
            echo "Error: Failed to get tab title for session ID $session_id"
            continue
        fi

        if [[ -n "${COMPONENTS[$tab_title]}" ]]; then
            # Extraer el nombre del ejecutable del comando
            command_parts=(${COMPONENTS[$tab_title]})
            last_part=${command_parts[-1]}
            process_name=$(basename "$last_part" | cut -d' ' -f1)

            # Kill the running process
            qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "killall -9 $process_name" 2>/dev/null
            
            # Remove the tab
            qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.removeSession "$session_id"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to remove tab with title $tab_title"
                continue
            fi
            echo "Removed tab with title: $tab_title"
        fi
    done
}

function start_agents() {
    echo "=== Starting agents ==="
    for TAB_NAME in "${!COMPONENTS[@]}"; do
        COMMAND="${COMPONENTS[$TAB_NAME]}"
        session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create new tab for $TAB_NAME"
            continue
        fi
        qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
        qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "$COMMAND"
        qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id
    done
}

# ===========================
# MAIN
# ===========================
if [ "$1" == "stop" ]; then
    clean_tabs
elif [ "$1" == "list" ]; then
    show_tabs
else
    clean_tabs
    start_agents
fi

echo "=== Script completado ==="