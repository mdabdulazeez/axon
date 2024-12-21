# Installation Guide for OpenBCI_GUI on Linux

This guide provides step-by-step instructions for installing and configuring OpenBCI_GUI on a Linux system.

## Useful Links
* [OpenBCI Downloads Page](https://openbci.com/downloads) - Download the OpenBCI GUI software
* [OpenBCI Documentation](https://docs.openbci.com/) - Complete documentation and guides
* [OpenBCI GUI Documentation](https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/) - Detailed GUI usage instructions

## Prerequisites

### 1. Java Runtime Environment
Ensure that the Java Runtime Environment (JRE) is installed on your system.

```bash
sudo apt-get install default-jre
```

Confirm the installation:
```bash
java -version
```

The output should display the installed Java version.

### 2. Unzip Tool
Ensure the `unzip` utility is available:

```bash
sudo apt-get install unzip
```

### 3. Administrator Privileges
You need `sudo` access to install and configure OpenBCI_GUI.

## Steps to Install OpenBCI_GUI

### 1. Download OpenBCI_GUI
* Visit the [OpenBCI Downloads Page](https://openbci.com/downloads) and download the latest Linux version of OpenBCI_GUI.


### 2. Extract the Downloaded File
* Navigate to the directory containing the downloaded file:

```bash
cd ~/Downloads
```

* Extract the file:

```bash
unzip openbcigui_v6.0.0-beta.1_linux64.zip
```

* A folder named `OpenBCI_GUI` will be extracted.

### 3. Move OpenBCI_GUI to /opt
* Move the extracted folder to the `/opt` directory for system-wide availability:

```bash
sudo mv OpenBCI_GUI /opt
```

## Create a Desktop Shortcut

### 1. Create the .desktop File
* Open a terminal and create a new `.desktop` file:

```bash
sudo nano /usr/share/applications/OpenBCI_GUI.desktop
```

### 2. Add the Following Content
Paste the following into the file:

```ini
[Desktop Entry]
Version=1.0
Name=OpenBCI GUI
Comment=Launch OpenBCI GUI
Exec=gnome-terminal -- bash -c "sudo /opt/OpenBCI_GUI/OpenBCI_GUI; echo 'Press Enter to close...' && read; exec bash"
Icon=/opt/OpenBCI_GUI/resources/OpenBCI_Icon.png
Terminal=true
Type=Application
Categories=Utility;
```

### 3. Set Permissions
* Make the `.desktop` file executable:

```bash
sudo chmod +x /usr/share/applications/OpenBCI_GUI.desktop
```

## Configure Sudo Access for OpenBCI_GUI

To avoid password prompts when launching the application:

### 1. Edit the sudoers file:
```bash
sudo visudo
```

### 2. Add the following line at the end:
```bash
<username> ALL=(ALL) NOPASSWD: /opt/OpenBCI_GUI/OpenBCI_GUI
```
Replace `<username>` with your Linux username.

## FTDI Buffer Configuration

This configuration has been verified to work with Ubuntu 18.04 and similar distributions.

### 1. Verify FTDI Driver Installation
Ensure that the FTDI driver is installed and you can connect to the Cyton.

### 2. Adjust Latency Timer
1. Locate the `latency_timer` file at:
```bash
/sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```
Note: `ttyUSB0` is the serial port name for the OpenBCI dongle.

2. Open the file and change the value from `16` to `1`:
```bash
sudo echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

### 3. Verify Configuration
Run the OpenBCI GUI or a BrainFlow binding to confirm smoother data collection.

## Launching OpenBCI_GUI

### 1. From the Desktop
* Locate the `OpenBCI GUI` shortcut in your applications menu and click it to launch.

### 2. From the Terminal
* You can also launch OpenBCI_GUI directly from the terminal:

```bash
sudo /opt/OpenBCI_GUI/OpenBCI_GUI
```

## Troubleshooting

### Audio Errors
If you encounter errors like `Unable to load audio files`:
* Ensure an audio device is connected and functional.
* Restart the GUI.

### Permissions Issues
If the application does not launch:
* Verify that the executable has the correct permissions:

```bash
sudo chmod +x /opt/OpenBCI_GUI/OpenBCI_GUI
```

### Logs and Debugging
To capture logs for debugging, use the terminal launch method and review the output messages for errors.

## Uninstallation

To remove OpenBCI_GUI:

### 1. Delete the installation directory:
```bash
sudo rm -rf /opt/OpenBCI_GUI
```

### 2. Remove the desktop shortcut:
```bash
sudo rm /usr/share/applications/OpenBCI_GUI.desktop
```