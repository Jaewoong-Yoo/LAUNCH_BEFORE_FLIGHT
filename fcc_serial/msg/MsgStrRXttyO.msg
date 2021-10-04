
uint8 HeaderA
uint8 HeaderB
uint8 HeaderC
uint8 HeaderD

float32 Cur_Time_sec

int8 UsedSBUSNum
int8 Mode_MotorAct
int8 Mode_FlightMode
int8 Mode_VehicleFlyingStatus

float32[6] ADC_volt

float32  GNSS_SatNum
float32  GNSS_hAcc_m
float32  GNSS_vAcc_m
float64 GNSS_iTOW_msec
float64[3] GNSS_ECEF_m
float64[3] GNSS_LLH_degdegm

float32 LidarPosDown_m
float32 LidarPos_m
float32 BaroPosDown_m

float32[2] FlowXY_mps
float32 SonarPosDown_m
float32 FlowQuality

float32 TXStickPosAil_usec
float32 TXStickPosEle_usec
float32 TXStickPosThr_usec
float32 TXStickPosRud_usec
float32[6] TX_CH_8_9_13_14_15_16_usec

float32 Cmd_R_dps
float32[3] Cur_AngularRate_dps
float32[3] Cmd_Att_deg
float32[3] Cur_Att_deg
float32[3] Cmd_LinAccAED_mpss
float32[3] Cur_LinAccAED_mpss
float32[3] Cmd_VelAED_mps
float32[3] Cur_VelAED_mps

float32 ThrottleTrim_usec
float32 ThrottleOut_usec
float32[8] MotorPWM_usec

float32[3] Cur_PosNED_m
float32[3] Cur_VelNED_mps


float32 GyroTemper_C
float32[3] GyroRawXYZ_dps
float32[3] SpeciAccRawXYZ_mpss
float32[3] MagRawXYZ_mG

uint16 SensorStatus

uint8 ChecksumA
uint8 ChecksumB

Header header

uint8 Agent_Number
