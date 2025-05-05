#!/bin/bash

# ===========================
# CONFIGURACIÓN DE COMPONENTES
# ===========================
declare -A COMPONENTS=(
    ["base_controller_agent"]="~/robocomp/components/robocomp-shadow/agents/base_controller_agent;bin/base_controller_agent etc/config"
    ["g2o_agent"]="~/robocomp/components/robocomp-shadow/agents/g2o_agent;src/g2o_agent.py etc/config"
    ["room_detector_bt"]="~/robocomp/components/robocomp-shadow/agents/room_detector_bt;bin/room_detector_bt etc/config"
    ["master_controller"]="~/robocomp/components/robocomp-shadow/agents/master_controller;src/master_controller.py etc/config"
    #["fridge_concept"]="~/robocomp/components/robocomp-shadow/agents/fridge_concept_python;src/fridge_concept etc/config"
)

function show_tabs()
{
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
function clean_tabs()
{
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
            # Kill the running process
            process_name=$(basename "${COMPONENTS[$tab_title]}" | cut -d' ' -f1)
            qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "killall -9 $process_name"
            if [ $? -ne 0 ]; then
                echo "Error: Failed to kill process $process_name for tab title $tab_title"
                continue
            fi

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

function start_agents
{
  echo "=== Starting agents ==="
      for TAB_NAME in "${!COMPONENTS[@]}"; do
          IFS=';' read -r DIRECTORY_PATH COMMAND <<< "${COMPONENTS[$TAB_NAME]}"
          session_id=$(qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.addSession)
          qdbus org.kde.yakuake /yakuake/tabs org.kde.yakuake.setTabTitle "$session_id" "$TAB_NAME"
          qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cd $DIRECTORY_PATH"
          qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "cmake -B build && make -C build -j32"
          qdbus org.kde.yakuake /yakuake/sessions runCommandInTerminal $session_id "$COMMAND"
          qdbus org.kde.yakuake /yakuake/sessions org.kde.yakuake.raiseSession $session_id
      done
}

# ===========================
# MAIN
# ===========================
if [ "$1" == "stop" ]; then
    clean_tabs
else
    show_tabs
    clean_tabs
    start_agents
fi

echo "=== Script completado ==="
