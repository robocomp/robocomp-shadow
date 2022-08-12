# aventuraconversacional
Intro to component here

For using omnidirectional microphone, execute in src folder:

git clone https://github.com/respeaker/usb_4_mic_array.git

## Configuration parameters
As any other component, *aventuraconversacional* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
EXAMPLE HERE
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <aventuraconversacional's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/aventuraconversacional config
```
