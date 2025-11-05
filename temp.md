```
current=$(uname -r | sed 's/-generic//')
sudo apt remove --purge -y $(dpkg --list | grep 'linux-image-[0-9]' | awk '{print $2}' | grep -v "$current")
sudo apt remove --purge -y $(dpkg --list | grep 'linux-modules-[0-9]' | awk '{print $2}' | grep -v "$current")
sudo apt remove --purge -y $(dpkg --list | grep 'linux-modules-extra-[0-9]' | awk '{print $2}' | grep -v "$current")
```
