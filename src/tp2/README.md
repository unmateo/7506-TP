# TP 2: _Organización de Gatos_

Link a competencia:https://www.kaggle.com/c/Inmuebles24/overview

## Preparación

Se usará la [API kaggle](https://github.com/Kaggle/kaggle-api). Para eso, hay que generar las credenciales y cargarlas en un archivo .env en el directorio raiz del repo.

## Organización

Creamos una clase Modelo, de la cual van a heredar los distintos modelos que preparemos. Esta clase tendrá un comportamiento default que en cada caso podrá sobreescribirse.

## Submit

kaggle competitions submit -c Inmuebles24 -f submission.csv -m "Message"
