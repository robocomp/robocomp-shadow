#!/bin/bash
# Scrip de auto arraque del shadow,debera tener permisos con sudo chmod +x autostart.sh y el rcnode e introducirlo en el crontab como:
# @reboot /home/robocomp/robocomp/components/robocomp-shadow/etc/autostart.sh

echo "Wait 20 second"
sleep 20
# Arrancamos rcnode tiene que tener permisos de ejecucion
echo "Start rcnode"
/home/robocomp/robocomp/tools/rcnode/rcnode.sh&

# Ubicamos en componente de la bateria y ejecutamos
echo "Start base"
cd /home/robocomp/robocomp/components/robocomp-robolab/components/hardware/battery/victron
src/victron.py etc/config_shadow&

# Ubicamos en componente de la base y ejecutamos
echo "Start base"
cd /home/robocomp/robocomp/components/robocomp-shadow/components/shadowbase
src/shadowbase.py etc/config&

# Ubicamos en componente del joystick y ejecutamos
cd /home/robocomp/robocomp/components/robocomp-robolab/components/hardware/external_control/joystickpublish
# Por fallo en identificar el mando se realizaran varios intentos siendo la ejecucion en primer plano
for i in {1..10};do
    sleep 10
    echo "Start joystickPublish"
    bin/JoystickPublish etc/config_shadow 
done
