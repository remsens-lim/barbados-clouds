# barbados-clouds

Object based cloud detection and classification for cloud radar measurments over the Barbados Cloud Observatory (BCO).
Warm clouds are defined as clouds with bases and tops below the height of the 0°C wet-bulb temperature isotherm. 
Trade wind clouds are clouds with a base below 1km (average height of the lifting condensation level over the BCO) and cloud top below the height of the 0°C wet-bulb temperature isotherm. 
Cold and mixed phase clouds are clouds with a cloud top above the height of the 0°C wet-bulb temperature isotherm. 

As input sources the **Cloudnet target classification** and **Cloudnet target categorization** products are mandatory.

![barbados-clouds example output](data/20211202_0-12UTC_cloudtypes.png)

# Installation
The package can be installed via pip:
```
 python -m pip install git@github.com:remsens-lim/barbados-clouds.git
```
