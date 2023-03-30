#!/bin/bash
# Scrip de auto arraque del shadow,debera tener permisos con sudo chmod +x autostart.sh y el rcnode e introducirlo en el crontab como:
# @reboot /home/robocomp/robocomp/components/robocomp-shadow/etc/autostart.sh


sleep 20
# Arrancamos rcnode tiene que tener permisos de ejecucion
/home/robocomp/robocomp/tools/rcnode/rcnode.sh &
# Ubicamos en componente de la base y ejecutamos
cd /home/robocomp/robocomp/components/robocomp-shadow/components/shadowbase
src/shadowbase.py etc/config &
# Ubicamos en componente del joystick y ejecutamos
cd /home/robocomp/robocomp/components/robocomp-robolab/components/hardware/external_control/joystickpublish
bin/JoystickPublish etc/config_shadow &
