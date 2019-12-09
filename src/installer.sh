# #!/bin/bash


function check_python_module_exists() {
	[ $(python -c "import $1; print($1.__version__)" 2>/dev/null |& grep -c "Error:") -eq 0 ]
}

function isntall_python_module() {
	if check_python_module_exists $2;
	then
		echo -e "$(tput setaf 3)already installed $(tput sgr 0)"
	else
		OUTPUT=$(sudo -H $3 install $1 2>&1)
		if check_python_module_exists $2;
		then
			echo -e "$(tput setaf 2)success $(tput sgr 0)"
		else
			echo -e "$(tput setaf 1)fail $(tput sgr 0)"
			echo -e ${OUTPUT}
		fi	
	fi
}


cd ~
mkdir git &>/dev/null
cd git
sudo apt-get update &>/dev/null

# install opencv
echo -e "$(tput setaf 6)INSTALLING OpenCV AND REQUIREMENTS...$(tput sgr 0)"

MODULES=('build-essential' 'cmake' 'git' 'libgtk2.0-dev' 'pkg-config'
			'libavcodec-dev' 'libavformat-dev' 'libswscale-dev'  'python3-dev'
			 'python3-numpy' 'libjpeg-dev' 'libpng-dev'
			 'python3-pip' 'python3-tk' 'python3-pyqt5' 'pyqt5-dev-tools')

for MODULE in ${MODULES[@]}
do
	echo -n "Installing $MODULE..."
	if [ $(dpkg-query -W -f='${Status}' $MODULE 2>/dev/null | grep -c "ok installed") -eq 0 ];
	then
   		OUTPUT=$(sudo apt-get -y install $MODULE  2>&1)
   		if [ $(dpkg-query -W -f='${Status}' $MODULE 2>/dev/null | grep -c "ok installed") -eq 0 ];
   		then
   			echo -e "$(tput setaf 1)fail$(tput sgr 0)"
   			echo -e $OUTPUT
   		else 
			echo -e "$(tput setaf 2)success$(tput sgr 0)"
   		fi
   	else 
   		echo -e "$(tput setaf 3)already installed$(tput sgr 0)"
	fi
done

PYTHON_MODULES=('tensorflow==1.14.0' 'opencv-python' 'dlib')

for MODULE in ${PYTHON_MODULES[@]}
do
	echo -n "Installing $MODULE..."
	isntall_python_module $MODULE $MODULE "pip2"	
done