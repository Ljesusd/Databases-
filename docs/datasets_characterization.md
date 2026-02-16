# Caracterizacion de datasets

Este resumen define la problematica por dataset y lista sujetos anomalos usando
solo trayectorias de marcadores o series de gaitEvents. No se calcularon joint
angles ni se usaron en graficas.

## Metodo de deteccion de anomalias
- Senales: trayectorias normalizadas de marcadores (C3D -> Eurobench) o series
  de gaitEvents.
- Metricas: rango (max-min), offset (RMS de la media), ruido (RMS de derivadas /
  rango).
- Outliers: z robusto > 3.5 (MAD) por sujeto.
- Ciclo incompleto: ausencia de trayectorias normalizadas o n_steps <= 5.

## 138_HealthyPiG
Problematica: dataset amplio con un solo ciclo por sujeto; algunas capturas no
logran normalizarse y aparecen offsets altos o ruido en trayectorias de
marcadores, lo que afecta comparabilidad inter-sujeto.

| Campo | Valor |
| --- | --- |
| Frecuencia de muestreo | 100 Hz (POINT rate C3D) |
| Sujetos | 138 grabados; 131 con trayectoria normalizada |
| Equipo | Vicon (C3D; Plug-in Gait) |
| Tipo de datos | Trayectorias de marcadores (RTHI/RKNE/RANK) |

| Condicion | Sujetos |
| --- | --- |
| gait | 138 |

| Tipo de anomalia | Sujetos |
| --- | --- |
| Ciclos incompletos (sin trayectoria normalizada) | SUBJ100, SUBJ118, SUBJ126, SUBJ131, SUBJ133, SUBJ19, SUBJ37 |
| Offset grande | SUBJ05, SUBJ06, SUBJ11, SUBJ117, SUBJ24, SUBJ28, SUBJ42, SUBJ60, SUBJ61, SUBJ68, SUBJ71, SUBJ87, SUBJ91, SUBJ94, SUBJ99 |
| Ruido / graficas irregulares | SUBJ105, SUBJ114, SUBJ119, SUBJ12, SUBJ120, SUBJ123, SUBJ127, SUBJ128, SUBJ134, SUBJ138, SUBJ21, SUBJ35, SUBJ48, SUBJ59, SUBJ62, SUBJ64, SUBJ66, SUBJ72, SUBJ92, SUBJ95 |

## Multisensor human gait dataset
Problematica: multiples sesiones por usuario (day1/day2) y dataset
multisensor; la variabilidad inter-sesion introduce ruido en algunos usuarios.

| Campo | Valor |
| --- | --- |
| Frecuencia de muestreo | 100 Hz (POINT rate C3D) |
| Sujetos | 25 usuarios |
| Equipo | Qualisys (QTM C3D); IMUs disponibles pero no usados |
| Tipo de datos | Trayectorias de marcadores (RTHI/RKNE/RANK) |

| Condicion | Sujetos |
| --- | --- |
| day1 | 25 |
| day2 | 25 |

| Tipo de anomalia | Sujetos |
| --- | --- |
| Ruido / graficas irregulares | user09 |

## Human gait and other movements
Problematica: n pequeno de sujetos y condiciones; 2minwalk no esta presente
en todos los sujetos. No hay fabricante reportado en los metadatos C3D.

| Campo | Valor |
| --- | --- |
| Frecuencia de muestreo | 100 Hz (POINT rate C3D) |
| Sujetos | 6 |
| Equipo | Sistema optico C3D (fabricante no indicado) |
| Tipo de datos | Trayectorias de marcadores (Eurobench) |

| Condicion | Sujetos |
| --- | --- |
| gait | 6 |
| 2minwalk | 5 |

| Tipo de anomalia | Sujetos |
| --- | --- |
| No se detectaron outliers con trayectorias | - |

## Zenodo_5668164 (treadmill + IMU + force platform)
Problematica: pocos sujetos y varias condiciones con exoesqueleto/treadmill;
algunas pruebas tienen muy pocos pasos o datos faltantes en gaitEvents.

| Campo | Valor |
| --- | --- |
| Frecuencia de muestreo | 300 Hz (gaitEvents time) |
| Sujetos | 3 |
| Equipo | Treadmill, plataforma de fuerza (COP/GRF) e IMUs; exo en condiciones Exo |
| Tipo de datos | gaitEvents.csv + platformData.csv |

| Condicion | Sujetos |
| --- | --- |
| Treadmill | 1 |
| Treadmill_1 | 1 |
| Treadmill_2 | 1 |
| Treadmill_Exo | 1 |
| Treadmill_noExo | 1 |

| Tipo de anomalia | Sujetos |
| --- | --- |
| Ciclos incompletos (n_steps <= 5 en al menos un trial) | SUBJECT_1, SUBJECT_3, SUBJECT_4 |
| Datos faltantes / NaNs | SUBJECT_1, SUBJECT_4 |
| Ruido / graficas irregulares | SUBJECT_4 |

## Running Injury Clinic gait dataset (1798)
Problematica: dataset grande con mezcla de walking/running; requiere limpieza
de metadatos de lesiones y no se ha corrido aun analisis de anomalias.

| Campo | Valor |
| --- | --- |
| Frecuencia de muestreo | Variable; se reporta por archivo (hz_w, hz_r) |
| Sujetos | 1798 |
| Equipo | Vicon (Bonita o MX3+, 3 u 8 camaras) |
| Tipo de datos | JSON con marcadores 3D (joints/neutral/walking/running) |

| Condicion | Sujetos |
| --- | --- |
| walking | 1798 |
| running | 1798 |

| Tipo de anomalia | Sujetos |
| --- | --- |
| No se ha corrido deteccion | - |
