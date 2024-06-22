//**********File description begin************************
// This is the firmware code to program the micro-processor on the motherboard of DPE testing platform.
// The micro-processsor communicates with the external computer and configure the latch registers on the testing boards.
// File version: V1.30
// Authors: Miao Hu, Eirc Montgomery, Yunning Li, Wenhao Song
// File creation date: Dec/2/2015
// Last update date:  Jun/19/2018
//**********File description end**************************

//**********Log file begin*******************************
// Aug/17/2018: V1.32: batch_conv2d and batch_conv2dlstm with repeated vertical DPs
// Aug/01/2018: V1.31: batch_conv2d and batch_conv2dlstm with dynamic input scaling.
// Jun/20/2018: V1.30: Convolutional 2D LSTM. Command 41.
// Jun/19/2018: V1.30: Convolution 2D. Command 40. Wenhao's idea to let the MCU to do the strides internally.
// May/20/2017: V1.28: Optimize the DAC writing. Use DAC's update_all command to achieve faster arbitrary wave generation
// May/18/2017: V1.27: Add generate arbitrary wave function as command 19
// Apr/24/2017: V1.26: modify TIA active and connect to device sequence that help avoid a short negtive pulse applied on TEs
// Apr/17/2017: V1.25d: modify the TIAs sequences which compatible with Umass measurement system.
// Sep/12/2016: V1.24d: modify the code to support reading/programing upper 64x64 array or lower 64x64 array with batch read, set and reset.
// Aug/30/2016: V1.24: Add batch slow set and slow reset in the code to reduce set coupling effect.
// Aug/26/2016: After fixed bugs in 1.22, in 1.23 add DPE read function and single device feedback programming function (not yet complete).
// Aug/26/2016: Fixed minor bugs on selector voltage settings.
// Aug/23/2016: Fixed the command 15 MPU write floating issue; Also removed the legacy COL_DAC_SPAN limitation on column selector votlage inputs.
// July/27/2016: Finally find the bug: it is the address error of oneline DAC setup for row7to15. In the oneline mode, Line address does not need +8.
// July/20/2016: try to debug batch set and reset functions.
// June/21/2016: Version 1.19 does not support 128x64 crossbar. It adds slow write function as command 15.
// June/21/2016: Version 1.18 supports 128x64 crossbar, but has no batch_set & batch_reset functions.
// June/17/2016: CHange the DAC !CLR signal settings in column safe start and row safe start to put DAC outputs quickly to zero value.
// June/13/2016: Change array size back to 64x64, the MCU could not support "data set larger than 64kB". It means some tricks on saving variables must be applied to extend to 128x64. Add commend 14 to grond all rows and cols.
// May/31/2016: Change maximum array variable from 64x32 to 128x64 to support full array testing.
// May/27/2016: Now use COL_DAC_SPAN for only column output span instead of all outputs, add COL_SEL_SPAN, COL_SEL_BATCH_SPAN for selector output span controls. Now COL_SEL_SPAN and COL_SEL_BATCH_SPAN are fixed to 10 in the firmware.
// May/25/2016: Code cleaning.
// May/23/2016: Fix the golbal/local variable issue.
// May/20/2016: Put in safe start functions for batch set and batch reset. So the rows and columns are grounded in the end.
// May/11/2016: Final reviewed for testing on real arrays.
// May/9/2016: Fixed a lot of MY_COUNTER bugs in the batch set and batch reset. It appears that all the variables are globally shareded. Needs to be very careful.
// April/25/2016: Try to fix the gate voltage bug in batch read/set/reest. The bug is fixed by config the DAC voltage span back to +/-5 before selector configuration.
// March/18/2016: Add Batch reset as opintion 12. Fixed some bugs in Batch set of version 1.11
// March/07/2016: Add Batch set as opintion 11.
// March/02/2016: fix small bugs in fast read mode output format. Fix batch read comunication issue.
// March/01/2016: Remove the column board 0 and 7 workaround; Also remove the serial.flush() at the beginning of the loop, which may cause data lose.
// Feb/23/2016: Remove all 1 microsecond delays and add batch read mode.
// Feb/19/2016: Checking bugs for possible comunication error. Optimize firmware performance by removing unnecessary delays in basic functions; Modified seperate update in col_dac_span and row_dac_span by adding setup for dac_voltage, sel_voltage, pulse_width control after.
// Feb/18/2016: Minor change to the version running in the lab. Removed auto-all grounding at this moment. Currently, 1.09b is the stable version currently running. More tests are needed to evaluate its performance to direct further improvement.
// Feb/11/2016: Add auto-all grounding before and after read.
// Feb/10/2016: Change slow pulse settings to use slow path, the pulse width is controlled by the delay btween fliping ROW_MUX_AX.
// Feb/09/2016: fix the high low switch problem.
// Feb/08/2016: Add slow pulse read/write mode.
// Feb/02/2016: Change vconst to dynamic value which changes with different DAC_SPAN.
// Feb/02/2016: Fixed vconst value.
// Feb/01/2016: Change column board and row board's DAC span options from 0 to 5V, 0 to 10V to +/-5V, +/- 10V; Extend the MPU pulse from 3us to 20us; version 1.05b put in the offset fix.
// Jan/30/2016: Modified by John Pual: change braud rate to 115.2k, set to default silent mode.
// Jan/27/2016: Fully tested, put the code into silent mode.
// Jan/20/2016: Add some println lines for check; Add option 6 to print out all current configurations.
// Jan/19/2016: Make change according to John Paul's suggestion.1: remove "go" at all sub-levels; 2, add "enter 1024 to return from any sub-level to top level" function.
// Jan/15/2016: Double checked and V1.0 is ready. All firmware now works correctly, the pulse width still needs fine tuning for accurate ADC reading.
// Jan/14/2016: Add seperate update; modified interfacel; fixed some bugs; added "go" flag for MATLAB sync.
// Jan/13/2016: Add flag to enable/disable serial.println output; working on partial update (Column board is done);
// Jan/12/2016: Add gate(select) control.
// Jan/7/2016: V0.171 added protection for Hi/Low_V_Sw to avoid too high voltage span damage the switch.
// Jan/7/2016: Can fully configure every parameters in the system; support MATLAB interface update.
// Jan/6/2016: Updated code for sample-and-hold pulse width and ADC-convst pulse width; fixed some bugs; slightly changed interface for future development.
// Dec/15/2015: Debug, the interface part works correctly.
// Dec/14/2015: First complete draft. Add commend line interface in the loop to re-configure all parameters while running.
// Dec/11/2015: Moduleized most codes.
// Dec/7/2015: Eric tested it and added codes for LCD.
// Dec/4/2015: Fixed some bugs.
// Dec/3/2015: Re-orginzed firmware fire; Addresses are described by array type; Send to Eric to check its functionality.
// Dec/2/2015: The firmware file is created from Eric's testing code.
//**********Log file end**********************************

//**********Firmware structure description begin************************
// Firmware functions are described with levels from 0 to 5. Level 0 is the lowest level settings and functions, Level 5 is the highest-level functions.
// Note: low level function should never use high level functions.
// Level 0 function: physical pin, alias, variables and constants settings.
// Level 1 function: basic column board, row board and latch address selecting; basic DAC, ADC and latch configuration.
// Level 2 function: safe-mode functions to prevent damaging the board.
// Level 3 function: column/row mux settings, TIA settings, pulse settings.
// Level 4 function: Testing programs and monitor program.
// Level 5 function: Device programming functions and DPE computing functions.
//**********Firmware structure description end**************************

//**********Firmware structure begin************************
// Firmware functions are described with levels from 0 to 5. Level 0 is the lowest level settings and functions, Level 5 is the highest-level functions.
// Level 0 function: physical pin, alias, variables and constants settings; LDC pin settings;
// Level 1 function: select_column_board_X(), select_row_board_X(), select_latch_BX(), X is from 1 to 8; select_DAC(), deselect_DAC(), pulse_the_latch().
// Level 2 function: column_safe_start(), row_safe_start(),
// Level 3 function: setup()
// Level 4 function:
// Level 5 function:
//**********Firmware structure end**************************

//Library files
// This file tests the behaviour of ADC in DPE project.
// The analog input is connected to the input of Sample and hold circuit of the PCB board, then the signal is passed to TIA and received by ADC.
//************************************
// The file functions in three steps:
// Step 1: Configure latchs to initilize the circuit and stay ready for reading out signals.
// Step 2: When the MPU tells the circuit to start sending signal (to crossbars), MPU_PULSE is set HIGH, then the signals are read out and hold at the ADC outputs.
// Step 3: Configure latchs to read signals from ADCs and convert to actual input value.
//************************************
// NOTE: in this file, all items are counted and labelled from 0.
//************************************

// #include <p32xxxx.h>
// The next few lines setup usage of the LCD Display
#include <LiquidCrystal.h>
// Declare pins (Arduino Names) used by LiquidCrystal library
LiquidCrystal lcd(54, 55, 56, 57, 58, 59);
// Pins 54-59 are the first six pins of Port B

//******************Level 0 function Begin******************************************

//pulse used to start reads - pulse length and timing of Pulse Trigger,
//  Sample/Hold, and ConvSt determined by digital trimpots
int MPU_PULSE = 70;

//pulse used for long writes - pulse length controlled
//by switching appropriate ADG1404 output MUX between GND and Slow_Out
int TRIG = 22;
int SLOW_PULSE = 71;
int SLOW_PULSE_WIDTH = 10;

//signal sent from ADC back to MPU to signal End of Conversion
int ADC_EOC = 12;

//four board address pins on the selected board, selecting at a maximum 8 column boards and 8 row boards.
int BOARD_ADDR[4] = {83, 84, 82, 28};

//next 2 lines can be used to disable latch inputs.
//I've been using NOT_BOARD_OE[0] as a "Master Enable" for latches
//If large numbers of Row and Column Boards are necessary, we could
//add multiple motherboards without MPU's and use these two signals
//to address different banks of boards.
int NOT_BOARD_OE[2] = {80, 81};

// Globally enable all latchs on the selected board.
int NOT_GLOBAL_LATCH_OE = 79;

//latch address pins on the selected board, selecting latch 0 to latch7.
int LATCH_ADDR[3] = {13, 72, 73};

//DB0-DB15 return measurements taken by ADC's
int DB[16] = {3, 5, 6, 9, 10, 39, 47, 77, 48, 74, 38, 49, 8, 76, 19, 18};

//RB0-RB7 are main data bus to arrays of latches which control everything
int RB[8] = {37, 36, 35, 34, 33, 32, 31, 30};

//******************************************************************************
//This section is a list of all of the alias values needed to write to individual
//controls on the different latches. Probably a very wasteful way of doing it since
//it uses a byte for each alias. (Miao, do you know a better way???  :^)

//First for the columnn boards....

// Column MUX configuration table:
// States:     Float Float Float Float Ground Slow_pulse Read Fast_pulse
// COL_MUX_A0:   0     1     0     1      0       1        0      1
// COL_MUX_A1:   0     0     1     1      0       0        1      1
// COL_MUX_EN:   0     0     0     0      1       1        1      1

//Latch B0 - Enable Pins for ADG1404 MUX attached to Column Line
int COL_MUX_EN = 255; // 1111 1111

//Latch B1 - Address 0 lines for ADG1404 MUX attached to Column Line
int COL_MUX_A0 = 0;
//int COL_MUX_A0_SLOW = 0;

//Latch B2 - Address 1 lines for ADG1404 MUX attached to Column Line. In testing, we enable all channels to read mode.
int COL_MUX_A1 = 0;
//int COL_MUX_A1_SLOW = 0;

//Latch B3 - Some Spares and Control Lines for Column Board Pulse Shaping
int COL_REG_SPARE_0  = RB[7];
int COL_REG_SPARE_1  = RB[6];
int COL_NOT_ADC_SET = RB[5];//Chip Select for Digital Trimpot Controlling ADC Convert Start Timing
int COL_DIGITAL_POTS_SCLK = RB[4];
int COL_NOT_PW_SET = RB[3];//Chip Select for Digital Trimpot Controlling Read Pulse Length
int COL_DIGITAL_POTS_DIN = RB[2];
int COL_NOT_SAMPLE_HOLD_SET = RB[1];//Chip Select for Digital Trimpot Controlling Sample and Hold Timing
int COL_HI_LOW_V_SW = RB[0];//Controls path taken by pulses sent to array.

//High selects the High Voltage Path (>5V)
//Low selects the Low Voltage Path (<5V)

//Latch B4 - Address 0 Lines for TIA gain range selection
int TIA_A0 = 0; //RB

//Latch B5 - Address 1 Lines for TIA gain range selection
int TIA_A1 = 0; //RB

//Latch B6 - Some More Spares and the ADC Control Lines
int COL_REG_SPARE_2 = RB[7];
int COL_REG_SPARE_3 = RB[6];
int COL_REG_SPARE_4 = RB[5];
int COL_REG_SPARE_5 = RB[4];
int COL_REG_SPARE_6 = RB[3];
int ADC_NOT_READ    = RB[2];
int ADC_NOT_CS      = RB[1];
int ADC_NOT_WRITE   = RB[0];

//Latch B7 - A Couple More Spares and the Column DAC Control Lines
int COL_REG_SPARE_7 = RB[7];
int COL_DAC_NOT_LDAC = RB[6];
int COL_DAC_NOT_CS   = RB[5];
int COL_DAC_SCLK     = RB[4];
int COL_REG_SPARE_8  = RB[3];
int COL_DAC_SDI      = RB[2];
int COL_DAC_NOT_TGP  = RB[1];
int COL_DAC_NOT_CLR  = RB[0];

//Now for the Row Boards

// Row MUX configuration table:
// States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
// ROW_MUX_A0:   0     1     0     1      0       1        0      1
// ROW_MUX_A1:   0     0     1     1      0       0        1      1
// ROW_MUX_EN:   0     0     0     0      1       1        1      1
int RB_NUMBER = 0; // row board number
int ROW_MUX_EN_0to7 = 0; // latch B0. enable channel 1.
int ROW_MUX_A0_0to7 = 0; //Latch B1 - Address 0 lines for ADG1404 MUX attached to Row Line.
int ROW_MUX_A1_0to7 = 0; //Latch B2 - Address 1 lines for ADG1404 MUX attached to Row Line.

//int ROW_MUX_A0_0to7_SLOW = 0; //Latch B1
//int ROW_MUX_A1_0to7_SLOW = 0; //Latch B2

//Latch B3 - Some Spares and Control Lines for Row Board Pulse Shaping
int ROW_SPARE_0  = RB[7];
int ROW_SPARE_1  = RB[6];
int ROW_SPARE_2  = RB[5];
int ROW_DIGITAL_POTS_SCLK = RB[4];
int ROW_NOT_PW_SET = RB[3];
int ROW_DIGITAL_POTS_DIN = RB[2];
int ROW_SPARE_3  = RB[1];
int ROW_HI_LOW_V_SW = RB[0];

int ROW_MUX_EN_8to15 = 0 ; //Latch B4 - More MUX Enables
int ROW_MUX_A0_8to15 = 0; //Latch B5 - More MUX Address Lines
int ROW_MUX_A1_8to15 = 0; //Latch B6 - More MUX Address Lines

//int ROW_MUX_A0_8to15_SLOW = 0; //Latch B5 - More MUX Address Lines
//int ROW_MUX_A1_8to15_SLOW = 0; //Latch B6 - More MUX Address Lines
//int ROW_MUX_EN_0to7_SLOW = 0;
//int ROW_MUX_EN_8to15_SLOW = 0;
//int COL_MUX_EN_SLOW = 0;
//Latch B7 - Two More Spares and ROW DAC Control Lines
int ROW_SPARE_4      = RB[7];
int ROW_DAC_NOT_LDAC = RB[6];
int ROW_DAC_NOT_CS   = RB[5];
int ROW_DAC_SCLK     = RB[4];
int ROW_SPARE_5      = RB[3];
int ROW_DAC_SDI      = RB[2];
int ROW_DAC_NOT_TGP  = RB[1];
int ROW_DAC_NOT_CLR  = RB[0];

// Variables and constants settings.
int Serial_println_ON = 0; // flag to enable or disable serial.println functions. Disable it to make it faster for MATLAB processing.

int CB_NUMBER = 0; // column board number
int COL_PULSE_WIDTH = 0; // The column board DAC pulse width (if DACs are enabled).
int SAMPLE_AND_HOLD = 0; // Sample and hold time delay.
int ADC_CONVST = 20; // ADC convst time delay.


int COL_DAC_SPAN = 5; // DAC input voltage span for all 8 column channels;
int COL_SEL_SPAN = 10; // DAC intput voltage span for all 8 selector channels;

float COL_DAC_VOLTAGE[8] = {1, 1, 1, 1, 1, 1, 1, 1}; // DAC input voltage values for all 8 column channels;
float COL_SEL_VOLTAGE[8] = {1, 1, 1, 1, 1, 1, 1, 1}; // DAC input voltages values for all 8 select channels;
int ROW_PULSE_WIDTH = 200; // The row board read pulse width.
int ROW_DAC_SPAN = 5; // DAC input voltage values for all 16 row channels;
float ROW_DAC_VOLTAGE_0to7[8] = {1, 1, 1, 1, 1, 1, 1, 1};
float ROW_DAC_VOLTAGE_8to15[8] = {1, 1, 1, 1, 1, 1, 1, 1};// DAC input voltage values for all 16 row channels;
//Latch B0 - Enable Pins for ADG1404 MUX attached to Row Line

char str1[24] = "Hewlett Packard Labs"; // string to display on the first row of the LCD.
char str2[24] = "Dot Product Engine"; // string to display on the first row of the LCD.
int ADC_READ_VALUE[8]; // Read out values from the ADC.

// All batch read/write variables:

float V_BATCH_READ = 0;
float V_BATCH_GATE = 0;
int TIA_BATCH_GAIN = 2;
int ROW_BATCH_PULSE_WIDTH = 1000;
int COL_BATCH_PULSE_WIDTH = 1000;
int COL_BATCH_SH_DELAY = 900;
int COL_BATCH_AD_CONVST_DELAY = 1010;
int NUM_COL_BOARDS = 0;
int NUM_ROW_BOARDS = 0;

int TOTAL_ROWS = 128;
int TOTAL_COLS = 64;

int CB_BATCH_NUMBER = 0;
int COL_DAC_BATCH_SPAN = 5;//TODO It is meaningless to maintain a global SPAN because:1.If there is only one consistant SPAN, it will be received during each operation;2.If not(e.g.,operating on different subarrays, it cann't be saved in one in value, theoretically each channel need one so it needs a int[64].
//Now that in practice we always keep one consistent SPAN in one operation(command) and always pass this parameter, it would be totally fine to receive a local value. Batch DPE is a special case here.
int COL_SEL_BATCH_SPAN = 10;
float COL_DAC_BATCH_VOLTAGE[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float COL_DAC_BATCH_VOLTAGE_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float COL_SEL_BATCH_VOLTAGE_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};
int TIA_BATCH_READ_A0;
int TIA_BATCH_READ_A1;

int RB_BATCH_NUMBER = 0;
int ROW_DAC_BATCH_SPAN = 5;
float ROW_DAC_BATCH_VOLTAGE_0to7[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float ROW_DAC_BATCH_VOLTAGE_8to15[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float ROW_DAC_BATCH_VOLTAGE_0to7_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};
float ROW_DAC_BATCH_VOLTAGE_8to15_ZERO[8] = {0, 0, 0, 0, 0, 0, 0, 0};

float V_BATCH_SET[128][64];
float V_BATCH_SET_GATE[128][64];
// int bottom_flag = 0;

// Simple 1T1R memristor model.
int MODEL_ON = 0;

//******************Level 0 function End******************************************


//******************Level 1 function Begin****************************************

void select_column_board(int x) {
  digitalWrite(BOARD_ADDR[0], bitRead(x,0));
  digitalWrite(BOARD_ADDR[1], bitRead(x,1));
  digitalWrite(BOARD_ADDR[2], bitRead(x,2));
  digitalWrite(BOARD_ADDR[3], LOW);
  delayMicroseconds(1);
}

// QUESTION: why for the row boards the BOARD_ADDR[3] is always HIGH?
void select_row_board(int x) {
  digitalWrite(BOARD_ADDR[0], bitRead(x,0));
  digitalWrite(BOARD_ADDR[1], bitRead(x,1));
  digitalWrite(BOARD_ADDR[2], bitRead(x,2));
  digitalWrite(BOARD_ADDR[3], HIGH);
  delayMicroseconds(1);
}

void select_latch(int x) {
  digitalWrite(LATCH_ADDR[0], bitRead(x,0));
  digitalWrite(LATCH_ADDR[1], bitRead(x,1));
  digitalWrite(LATCH_ADDR[2], bitRead(x,2));
  delayMicroseconds(1);
}

void send_to_DAC(int nbits, int data) { //send the lowest nbits data to DAC from high bits to low bits.
  //assumes you have already selected a board and have selected latch B7
  for (int i = nbits - 1; i >= 0; i--) {
    PORTE = B01000011 | bitRead(data, i) << 2; //71;//Din at S[2], lower clock at S[4]
    PORTE = B01010011 | bitRead(data, i) << 2; //87;//raise clock at S[4], data in at clock raising edge
  }
}

void select_the_DAC() { //assumes you have already selected a board and have selected latch B7
  PORTE = B01000011;//67;//DAC_!CS lowered //S[1] is TGP
  delayMicroseconds(1);
}

void deselect_the_DAC() {
  PORTE = B01100011;//99;//back to default safe mode - assumes you have already selected a board and have selected latch B7
  delayMicroseconds(1);
}

void pulse_the_latch() { //Taking NOT_BOARD_OE[0] low, then high latches data on control bus into selected latch
  digitalWrite(NOT_BOARD_OE[0], LOW);
  //delayMicroseconds(1);
  digitalWrite(NOT_BOARD_OE[0], HIGH);
  delayMicroseconds(1);
}

//******************Level 1 function End****************************************

//******************Level 2 function Begin******************************************

void column_safe_start() {//TODO: Does this sequence different from select-set-pulse matters?
  // Before enables MUX, it is safer to turn them all off and configure other latches first. (Not applied.)
  //digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable Latch Enables
  //delayMicroseconds(1);
	send_to_latch(7, B01100010);//RB = 98, Spare8=0,DAC_!LDAC=1,DAC_!CS=1,DAC_SCLK=0,Spare=0,DAC_SDI=0,DAC_!TGP=1,DAC_!CLR=0
  send_to_latch(7, B01100011);//RB = 99, Spare8=0,DAC_!LDAC=1,DAC_!CS=1,DAC_SCLK=0,Spare=0,DAC_SDI=0,DAC_!TGP=1,DAC_!CLR=1
  send_to_latch(0, 0);// MUX_EN = [1111 1111], Enables all MUX on all columns. TODO: maybe can delete
  send_to_latch(1, 0);////HM_CHANGE, COL_MUX_A0 = [0000 0000] to point all columns at GND
  send_to_latch(2, 0);// COL_MUX_A1 = [0000 0000] to Point all columns at GND
  send_to_latch(0, 255);// MUX_EN = 255, Enables all MUX on all columns
  // Change Hi/Low_SW = 0, now remember if want to write with voltage higher than 5V, its DAC voltage must be set <5V before all grounding!!!
  send_to_latch(3, 42);// RB = [0010 1010] Spare1=0,Spare2=0,!ADCSet=1,Digital_Pots_SCLK=0,!PwSet=1,Digital_Pots_Din=0,!S/H=1,Hi/Low_V_Sw=0
  send_to_latch(4, 0);//TIA Gains All Set Low
  send_to_latch(5, 0);//TIA Gains All Set Low
  send_to_latch(6, 7); //RB = [0000 0111], Spare3=0,Spare4=0,Spare5=0,Spare6=0,Spare7=0,ADC_!RD=1,ADC_CS=1,ADC_!WR=1

}

void row_safe_start() {
  send_to_latch(7, 98);//Spare8=0,DAC_!LDAC=1,DAC_!CS=1,DAC_SCLK=0,Spare=0,DAC_SDI=0,DAC_!TGP=1,DAC_!CLR=0
  send_to_latch(7, 99);//Spare8=0,DAC_!LDAC=1,DAC_!CS=1,DAC_SCLK=0,Spare=0,DAC_SDI=0,DAC_!TGP=1,DAC_!CLR=1
  send_to_latch(0, 0);//Enables 1-8 On, same TODO
  send_to_latch(1, 0);//HM_CHANGE, ROW_MUX_A0 = [0000 0000] to point All rows at GND
  send_to_latch(2, 0);//Point Rows 1-8 at GND
  send_to_latch(0, 255);//Enables 1-8 On
  send_to_latch(3, 8);//Spare1=0,Spare2=0,Spare3=0,Digital_Pots_SCLK=0,!PwSet=1,Digital_Pots_Din=0,Spare4=0,Hi/Low_V_Sw=0
  send_to_latch(4, 0);//Enables rows 9-16 On
  send_to_latch(5, 0); //HM_CHNAGE: Point Rows 9-16 at GND
  send_to_latch(6, 0);//Point Rows 9-16 at GND
  send_to_latch(4, 255);//Enables rows 9-16 On
}
void safeStartAllRowsAndColumns() {
	//Set All Board Registers to "Safe" Values *******************************************
	for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
		select_row_board(MY_COUNTER0);
		row_safe_start();
	}
	for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
		select_column_board(MY_COUNTER0);
		column_safe_start();
	}
}
//******************Level 2 function End********************************************

//******************Level 3 function Begin******************************************
int PORTE_conversion_for_single_channel(int current_number) {
  return 1 << current_number;
}

// LCD display function
void lcd_display(char str1[], char str2[]) {// we don't care
  //set up LCD's number of columns and rows
  lcd.begin(24, 2);
  //Print a message to the LCD
  //lcd.print("Hewlett Packard Labs");
  lcd.print(str1);
  //set cursor to column 0, line 1
  //(note: line 1 is the second row, since counting begins with 0);
  lcd.setCursor(0, 1);
  //lcd.print("Dot Product Engine");
  lcd.print(str2);
}

void send_data_serial(byte mask, int nbits, int data) { //send the lowest nbits data to some 8-bit serial port from high bits to low bits.
	//assume mask[4] is clock pin and mask[2] is data pin.
	//assumes you have already selected a board and have selected latch B3
	for (int i = nbits - 1; i >= 0; i--) {//maybe need delay
		PORTE = mask | bitRead(data, i) << 2; //35;//Din at S[2], lower clock at S[4]
		PORTE = mask | bitRead(data, i) << 2 | 1<<4; //51;//raise clock at S[4], data in at clock raising edge
	}
}
void setup_col_pulse_width(int COL_PULSE_WIDTH_TEMP, int CB_NUMBER_TEMP, int COL_DAC_SPAN_TEMP) {
  //Set Pulse Width on Column Board-begin*************************************************
  //Since we're trying to do a read, column board pulse width setting won't make any difference,
  //but we will set it to max to avoid switch noise in TIA.
  //When doing a read, the pulse width set on the row boards is what matters.
  //Still Pointing at the same Column Board
  select_column_board(CB_NUMBER_TEMP);
  select_latch(3);//B3 has controls for digital trimpots
// From Low bit(RB(0)) to high bit(RB(7)):
// 1 Sets Hi/Low_V_SW to HIGH (probably unnecessary precaution) 0 to LOW High is slow
// 1 Sets Sample and Hold Trimpot Chip Select High
// 0 Clears Din
// 0 Chip Selects Pulse Width Trimpot
// 0 Clears SCLK
// 1 Sets ADC ConvSt Trimpot Chip Select High
// 0 Spare 1 left low
// 0 Spare 2 left low
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow changes to B3
  delayMicroseconds(1);
  //First we set Pulse Width Trimpot
  send_data_serial(B00100011, 8, B0);
  send_data_serial(B00100011, 16, COL_PULSE_WIDTH_TEMP);
  send_data_serial(B00100011, 6, B0);
  //Raise Chip Select for Pulse Width Digital Trimpot to latch new values into it
  PORTE = 42 | (COL_DAC_SPAN_TEMP > 5);
  delayMicroseconds(1);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//disable latch inputs while we move to Row Board
  //Set Pulse Width on Column Board-end*************************************************
}

void setup_col_sample_and_hold(int SAMPLE_AND_HOLD_TEMP, int CB_NUMBER_TEMP, int COL_DAC_SPAN_TEMP) {
	//Lowers Sample and Hold Chip Select
  select_column_board(CB_NUMBER_TEMP);
  select_latch(3);//B3 has controls for digital trimpots
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow changes to B3
  delayMicroseconds(1);
  send_data_serial(B00101001, 8, B0);//41
  send_data_serial(B00101001, 16, SAMPLE_AND_HOLD_TEMP);
  send_data_serial(B00101001, 6, B0);
  //Raise Chip Select for Digital Trimpot to latch new values into it
  PORTE = 42 | (COL_DAC_SPAN_TEMP > 5);
  delayMicroseconds(1);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//disable latch inputs while we move to Row Board
}

void setup_col_adc_convst(int ADC_CONVST_TEMP, int CB_NUMBER_TEMP, int COL_DAC_SPAN_TEMP) {
	//Lowers ConvSt Chip Select
  select_column_board(CB_NUMBER_TEMP);
  select_latch(3);//B3 has controls for digital trimpots
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow changes to B3
  send_data_serial(B00001011, 8, B0);//11
  send_data_serial(B00001011, 16, ADC_CONVST_TEMP);
  send_data_serial(B00001011, 6, B0);
  //Raise Chip Select for Digital Trimpot to latch new values into it
  PORTE = 42 | (COL_DAC_SPAN_TEMP > 5); 
  delayMicroseconds(1);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//disable latch inputs while we move to Row Board
}

void setup_col_dac_span(int COL_DAC_SPAN_TEMP, int CB_NUMBER_TEMP) {//B0110 (Write Span to n;) TODO:why not use span to all nowthat you set all channels the same here
  select_column_board(CB_NUMBER_TEMP);
  select_latch(7);//for DAC
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  for (int DAC_address = 0; DAC_address < 8; DAC_address++) {
    //WARNING!! ALL DAC SUBROUTINES ASSUME YOU HAVE CHOSEN CORRECT BOARD AND LATCH!!!!!
  //for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
  //  int DAC_address = MY_COUNT1;
    select_the_DAC();
    //send command - 4 bits 1110: Write Span to All; 0110: Write Span to n;
    send_to_DAC(4, B0110);
    //send address - 4 bits (don't cares in this case since were using 'global' span set command)
    //send address - 4 bits (after version 1.16, we change Write Span to all to Write Span to n, so we need to enter line address now)
    send_to_DAC(4, DAC_address);
    //send span value - 16 bits (only last 3 hold span code)
    //Write Span Code: 000: 0 to 5; 001: 0 to 10; 010: -5 to 5; 011: -10 to 10; 100: -2.5 to 2.5;
    //Note: if you change the Span here, you must change the DAC voltage conversion functions as well!
    if (COL_DAC_SPAN_TEMP == 10) {
      send_to_DAC(16, B0011);
    } else { // means  -5~5
      send_to_DAC(16, B0010);
    }

    deselect_the_DAC();
  }
  //set the column board DAC span - end
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}
//B0110 (Write Span to n;)
//TODO In many cases, just need to write span to all
void setup_col_sel_span(int COL_SEL_SPAN_TEMP, int CB_NUMBER_TEMP) {
	select_column_board(CB_NUMBER_TEMP);
	select_latch(7);//for DAC
	digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
	for (int DAC_address = 8; DAC_address < 16; DAC_address++) {
	//for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
	//	int DAC_address = MY_COUNT1 + 8;
		select_the_DAC();
		send_to_DAC(4, B0110);//1110: Write Span to All; 0110: Write Span to n;
		send_to_DAC(4, DAC_address);
		//Note: if you change the Span here, you must change the DAC voltage conversion functions as well!
		if (COL_SEL_SPAN_TEMP == 10) {//Span Code : 000 : 0 to 5; 001: 0 to 10; 010: -5 to 5; 011: -10 to 10; 100: -2.5 to 2.5;
			send_to_DAC(16, B0011);
		}
		else {
			send_to_DAC(16, B0010);
		}
		deselect_the_DAC();
	}
	digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}
int float2raw(float voltage, int dacSpan) {//voltage conversion function for +/- voltage range. eg,32768/2 which should be 2.5V on 0-5V range TODO: to replace direct calculation elsewhere
	return int((voltage + (float)dacSpan) / ((float)dacSpan * 2) * (float)65536);
}
void setup_col_dac_voltage(float COL_DAC_VOLTAGE_TEMP[8], int CB_NUMBER_TEMP, int COL_DAC_SPAN_TEMP) {//B0011 (write code to n, update n)
  select_column_board(CB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  // Set up the column board DAC voltage on all 8 channels
  for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
    int DAC_address_bit = MY_COUNT1;
    int DAC_voltage_bit = int ((COL_DAC_VOLTAGE_TEMP[MY_COUNT1] + (float)COL_DAC_SPAN_TEMP) / ((float)COL_DAC_SPAN_TEMP * 2) * (float)65536); 
    send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
  }
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_col_dac_voltage_oneline(float COL_DAC_VOLTAGE_TEMP, int LINE_ADDRESS, int CB_NUMBER_TEMP, int COL_DAC_SPAN_TEMP) {//B0011
  select_column_board(CB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  // Set up the column board DAC voltage on all 8 channels
  int DAC_address_bit = LINE_ADDRESS;
  int DAC_voltage_bit = int ((COL_DAC_VOLTAGE_TEMP + (float)COL_DAC_SPAN_TEMP) / ((float)COL_DAC_SPAN_TEMP * 2) * (float)65536); //voltage conversion function for +/- voltage range.
  send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_col_sel_voltage(float COL_SEL_VOLTAGE_TEMP[8], int CB_NUMBER_TEMP, int COL_SEL_SPAN_TEMP) {//B0011
  select_column_board(CB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
    int DAC_address_bit = MY_COUNT1 + 8;
    int DAC_voltage_bit = int ((COL_SEL_VOLTAGE_TEMP[MY_COUNT1] + (float)COL_SEL_SPAN_TEMP) / ((float)COL_SEL_SPAN_TEMP * 2) * (float)65536); //voltage conversion function for +/- voltage range.
    send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
	
  }
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_col_sel_voltage_oneline(float COL_SEL_VOLTAGE_TEMP, int LINE_ADDRESS, int CB_NUMBER_TEMP, int COL_SEL_SPAN_TEMP) {//B0011
  select_column_board(CB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  int DAC_address_bit = LINE_ADDRESS + 8;
  int DAC_voltage_bit = int ((COL_SEL_VOLTAGE_TEMP + (float)COL_SEL_SPAN_TEMP) / ((float)COL_SEL_SPAN_TEMP * 2) * (float)65536);
  send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_row_pulse_width(int ROW_PULSE_WIDTH_TEMP, int RB_NUMBER_TEMP, int ROW_DAC_SPAN_TEMP) {//TODO to be read and adapted - the homemade function
  //int MY_COUNTER;
  select_row_board(RB_NUMBER_TEMP);
  delayMicroseconds(1);
  select_latch(3);//B3 has controls for digital trimpots
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow changes to B3

  if (ROW_DAC_SPAN_TEMP > 5) {
    PORTE = B00000001;//1
	//Sets Hi/Low_V_SW to high
    //Sets Sample and Hold Chip Select high
    //Sets Din low
    //Sets Pulse Width Chip Select high
    //Sets SCLK low
    //Lowers ConvSt Chip Select
    //Leaves COL_SPARE_1 and COL_SPARE_2 low
  } else {
    PORTE = B00000000;//0
	//Sets Hi/Low_V_SW to low,
    //Clears Reg Spare 4
    //Clears Din
    //Chip Selects Pulse Width Trimpot
    //Clears SCLK
    //Clears Reg Spare 1-3
  }
  delayMicroseconds(1);
  //send first 8 empty bits
  for (int MY_COUNTER = 0; MY_COUNTER < 8; MY_COUNTER++) {
    PORTE = B00010001;//17;//raise SCLK
    delayMicroseconds(1);
    PORTE = B00000001;//1;//lower clock
  }
  //Send Pulse Width  -- This is my homemade SPI Interface at work! :^)
  for (int MY_COUNTER = 0; MY_COUNTER < 16; MY_COUNTER++) {
    if (bitRead(ROW_PULSE_WIDTH_TEMP, 15) == 1) {
      PORTE = B00000101;//5;//Set Din to 1
      delayMicroseconds(1);
      PORTE = B00010101;//21;//Raise SCLK
      delayMicroseconds(1);
      PORTE = B00000101;//5;//Lower SCLK
      delayMicroseconds(1);
      PORTE = B00000101;//1;//Set Din back to 0
    } else {
      PORTE = B00010001;//17;//Raise SCLK
      delayMicroseconds(1);
      PORTE = B00000001;//1;//Lower SCLK
      delayMicroseconds(1);
    }
    ROW_PULSE_WIDTH_TEMP = ROW_PULSE_WIDTH_TEMP << 1;
  }
  //Send last 6 empty bits
  for (int MY_COUNTER = 0; MY_COUNTER < 6; MY_COUNTER++) {
    PORTE = 17;//Raise SCLK
    delayMicroseconds(1);
    PORTE = 1;//Lower SCLK
    delayMicroseconds(1);
  }

  //Raise Chip Select for Digital Trimpot to latch new values into it
  if (ROW_DAC_SPAN_TEMP > 5) {
    PORTE = 9;
  } else {
    PORTE = 8;
  }
  delayMicroseconds(1);

  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches
}

void setup_row_dac_span(int ROW_DAC_SPAN_TEMP, int RB_NUMBER_TEMP) {//B1110
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches

  //WARNING!! ALL DAC SUBROUTINES ASSUME YOU HAVE CHOSEN CORRECT BOARD AND LATCH!!!!!
  //set spans
  select_the_DAC();
  //send command - 4 bits
  send_to_DAC(4, B1110);
  //send address - 4 bits (don't cares in this case since were using 'global' span set command)
  send_to_DAC(4, B0000);
  //send span value - 16 bits (only last 3 hold span code)// Set S1 to 0 to enable 0~5V or 0~10V.// Set S1 to 1 to enable +/- 5V or +/- 10V
  if (ROW_DAC_SPAN_TEMP == 10) {
    send_to_DAC(16, B0011);
  } else { // means  that COL_DAC_SPAN == 5.
    send_to_DAC(16, B0010);
  }

  deselect_the_DAC();
  //set up DAC span - end
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_row0to7_dac_voltage(float ROW_DAC_VOLTAGE_0to7_TEMP[8], int RB_NUMBER_TEMP, int ROW_DAC_SPAN_TEMP) {//B0011
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
    int DAC_address_bit = MY_COUNT1;
    int DAC_voltage_bit = int ((ROW_DAC_VOLTAGE_0to7_TEMP[MY_COUNT1] + (float)ROW_DAC_SPAN_TEMP) / ((float)ROW_DAC_SPAN_TEMP * 2) * (float)65536);
    send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
  }
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_row_dac_voltage_oneline(float ROW_DAC_VOLTAGE_0to7_TEMP, int LINE_ADDRESS, int RB_NUMBER_TEMP, int ROW_DAC_SPAN_TEMP) {//B0011
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
//  int DAC_address_bit = LINE_ADDRESS;
  int DAC_voltage_bit = int ((ROW_DAC_VOLTAGE_0to7_TEMP + (float)ROW_DAC_SPAN_TEMP) / ((float)ROW_DAC_SPAN_TEMP * 2) * (float)65536);
  send_DAC_0011(LINE_ADDRESS, DAC_voltage_bit);
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_row8to15_dac_voltage(float ROW_DAC_VOLTAGE_8to15_TEMP[8], int RB_NUMBER_TEMP, int ROW_DAC_SPAN_TEMP) {//B0011
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  for (int MY_COUNT1 = 0; MY_COUNT1 < 8; MY_COUNT1++) {
    int DAC_address_bit = MY_COUNT1 + 8;
    int DAC_voltage_bit = int ((ROW_DAC_VOLTAGE_8to15_TEMP[MY_COUNT1] + (float)ROW_DAC_SPAN_TEMP) / ((float)ROW_DAC_SPAN_TEMP * 2) * (float)65536);
    send_DAC_0011(DAC_address_bit, DAC_voltage_bit);
  }
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void send_DAC_0011(int DAC_address_bit, int DAC_voltage_bit){//B0011 (write code to n, update n)
  select_the_DAC();
  //send command - 4 bits (write code to n, update n)
  send_to_DAC(4, B0011);
  //send address - 4 bits
  send_to_DAC(4, DAC_address_bit);
  //send value - 16 bits (sending 32768/2 which should be 2.5V on 0-5V range
  send_to_DAC(16, DAC_voltage_bit);
  deselect_the_DAC();
}
void send_to_latch(int iLatch, int data){
	select_latch(iLatch);
	PORTE = data;
	pulse_the_latch();
}
void send_to_latch_with_delay(int iLatch, int data, int tDelay){
	select_latch(iLatch);
	delayMicroseconds(tDelay);
	PORTE = data;
	delayMicroseconds(tDelay);
	pulse_the_latch();
}

//******************Level 3 function End********************************************

//******************Level 4 function Begin******************************************
// feedback single device tuning function.

void SingleDeviceTuning() {

}

void GroundingAllRowsAndColumns() {
  // grounding all rows and columns to discharge before next device.
  // int MY_COUNTER0;
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < NUM_COL_BOARDS; MY_COUNTER0++) {
    CB_BATCH_NUMBER = MY_COUNTER0;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in fast pulse mode: 0, 255, 255
    //Set-up Column MUX Enable********************************************

    //Set-up Column MUX_A0 and MUX_A1********************************************
	send_to_latch(1,0);//MUX A0 
	send_to_latch(2,0);//MUX A1 
	send_to_latch(0,255);//MUX Enable
	
  }

  for (int MY_COUNTER0 = 0; MY_COUNTER0 < NUM_ROW_BOARDS; MY_COUNTER0++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER0;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in grounding mode.
	send_to_latch(1,0);//Row A0
	send_to_latch(2,0);//Row A1
	send_to_latch(0,255);//Row Enable  ROW_MUX_EN_0to7
	send_to_latch(5,0);//Row A0
	send_to_latch(6,0);//Row A1 
	send_to_latch(4,255);//Row Enable  ROW_MUX_EN_8to15
	
  }
}

//******************Level 4 function End********************************************
//Main file starts, setup() and then loop().
// setup function for quick testing.
void setup() {
  //set up LCD's number of columns and rows
  lcd_display(str1, str2);
  Serial.begin(115200 * 4);
  //Serial.begin(9600);
  if (Serial_println_ON == 1) {
    Serial.println("DPE system initilizing...");
  }
  TRISA = 0;//Writes PORTA as output, contains latch pins
  TRISE = 0;//Writes PORTE (RB0-RB7) as output
  TRISG = 0;//Writes PORTG as output
  pinMode(TRIG, OUTPUT);
  //TRISD = 1;

  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 16; MY_COUNTER0++) {
    pinMode(DB[MY_COUNTER0], INPUT);
  }

  digitalWrite(NOT_BOARD_OE[0], HIGH); //Disable Master Latch Control
  digitalWrite(NOT_BOARD_OE[1], LOW);

  digitalWrite(NOT_GLOBAL_LATCH_OE, HIGH); // Disable all latches' output on the selected board.
  digitalWrite(SLOW_PULSE, LOW);
  digitalWrite(MPU_PULSE, LOW);
  digitalWrite(TRIG, LOW);

  safeStartAllRowsAndColumns();
  //Disable Master Latch Control
  digitalWrite(NOT_BOARD_OE[0], HIGH);
  delayMicroseconds(1);

  //Re-enable output latches now that safe values are set up
  digitalWrite(NOT_GLOBAL_LATCH_OE, LOW);

  // To save dataset space, turn off the model variables.
  // Initlize the memristor model

  if (Serial_println_ON == 1) {
    Serial.println("DPE system set-up complete.");
  }
}

void loop() {
  if (Serial_println_ON == 1) {
    Serial.println("1.24d_Nov-7-2016. 1 for column board update, 2 for row board update, 3 for seperate update, 4 for fast read, 5 for fast write, 6 for configuration, 7 to slow read, 8 to trigger interface display, 9 to exit, 10 to Batch read, 11 to Batch set, 12 to Batch reset.");
    Serial.println("13 to turn on memristor model, 14 to set all row and col DAC to 0 values then grounding all rows and cols, 15 for slow write. 16 for DPE read. At any sub-level, enter 1024 will return to top level.");
  }
  //Set-up begins

  /*
    // Use flush().
    // Not work as intended, actually it will cause double "go" issue.
    // serial.flush();
  */

  /*
    // Use this to empty the input buffer.
    // Not work as intended, it makes the MATLAB could not receive the "go" for the next command.
    while(Serial.available() > 0) {
    TEMP1 = Serial.parseInt();
    }
  */
  Serial.println("go");
  while (Serial.available() == 0) {}
  if (Serial.available()) {
    int swtch1 = Serial.parseInt();
    switch (swtch1) {
      case 1:
        configure_cols();
        break;
      case 2:
        configure_rows();
        break;
      case 3:
        seperate_update();
        break;
      case 4:
        fast_read();
        break;
      case 5://Send Pulse That Starts the Entire Process
        digitalWrite(MPU_PULSE, HIGH);
        delayMicroseconds(5);
        digitalWrite(MPU_PULSE, LOW);
        break;
      case 6:
        display_configuration();
        break;
      case 7:// slow read (MPU read)
        mpu_read();
        break;
      case 8:
        Serial_println_ON = !Serial_println_ON;
        Serial.print("Serial_println_ON is changed to ");
        Serial.println(Serial_println_ON);
        break;
      case 9:
        Serial.println("Stopping program. Reboot to start again.");
        digitalWrite(NOT_GLOBAL_LATCH_OE, HIGH);
        exit(0); //Exits program
        break;
      case 10: //batch read
        batch_read();
        break;
      case 11: //batch reset
        batch_reset();
        break;
      case 12: //batch set
        batch_set();
        break;
      case 13: //turn on/off memristor model.
        MODEL_ON = !MODEL_ON;
        Serial.print("MODEL_ON is changed to ");
        Serial.println(MODEL_ON);
        break;
      case 14: //Add commend 14 to grond all rows and columns.
        ground_all_rows_cols();
        break;
      case 15: // MPU write.
        mpu_write();
        break;
      case 16:
        dpe_read();
        break;
      case 17: //batch set slow
        batch_set_slow();
        break;
      case 18:
        Batch_Reset_Slow();
        break;
      case 19:
        Arbitrary_Wave();
        break;
      case 20:
        Batch_DPE();
        break;
      case 31:
        batch_reset_monovoltage();
        break;
      case 32:
        batch_set_row_voltage();
        break;
      case 33:
        batch_read_bin();
        break;
      case 34:
        batch_set_float();
        break;
    		case 40:
			batch_conv2d();
			break;
		case 41:
			batch_conv2dlstm();
			break;
		}
	}
}

int configure_cols() {//configure one column
  // Input all configurations for columns - begin
  // Begin with the column board
  if (Serial_println_ON == 1) {
    Serial.println("Enter the CB_NUMBER (0-7): ");
  }
  while (Serial.available() == 0) {}
  CB_NUMBER = Serial.parseInt();
  if (CB_NUMBER == 1024) return -1;
  CB_NUMBER = constrain(CB_NUMBER, 0, 7);
  if (Serial_println_ON == 1) {
    Serial.println(CB_NUMBER);
  }
  
  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_MUX_EN(0-255): ");
  }

  while (Serial.available() == 0) {}
  COL_MUX_EN = Serial.parseInt();
  if (COL_MUX_EN == 1024) return -1;
  COL_MUX_EN = constrain(COL_MUX_EN, 0, 255);
  if (Serial_println_ON == 1) {
    Serial.println(COL_MUX_EN);
  }
  
  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_MUX_A0(0-255): ");
  }
  while (Serial.available() == 0) {}
  COL_MUX_A0 = Serial.parseInt();
  if (COL_MUX_A0 == 1024) return -1;
  COL_MUX_A0 = constrain(COL_MUX_A0, 0 , 255);
  if (Serial_println_ON == 1) {
    Serial.println(COL_MUX_A0);
  }
  
  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_MUX_A1(0-255): ");
  }
  while (Serial.available() == 0) {}
  COL_MUX_A1 = Serial.parseInt();
  if (COL_MUX_A1 == 1024) return -1;
  COL_MUX_A1 = constrain(COL_MUX_A1, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_DAC_SPAN(5 or 10)");
  }
  while (Serial.available() == 0) {}
  COL_DAC_SPAN = Serial.parseInt();
  if (COL_DAC_SPAN == 1024) return -1;
  if (COL_DAC_SPAN > 7) {
    COL_DAC_SPAN = 10;
  } else {
    COL_DAC_SPAN = 5;
  }
  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_DAC_VOLTAGE[8], seprated by comma (+/- COL_DAC_SPAN): ");
  }

  while (Serial.available() == 0) {}
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    COL_DAC_VOLTAGE[MY_COUNTER0] = Serial.parseFloat();
    if (COL_DAC_VOLTAGE[MY_COUNTER0] == (float)1024) return -1;
    //COL_DAC_VOLTAGE[MY_COUNTER0] = constrain(COL_DAC_VOLTAGE[MY_COUNTER0], 0, COL_DAC_SPAN);
    COL_DAC_VOLTAGE[MY_COUNTER0] = constrain(COL_DAC_VOLTAGE[MY_COUNTER0], -COL_DAC_SPAN, COL_DAC_SPAN);
  }
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
      Serial.print(COL_DAC_VOLTAGE[MY_COUNTER0]);
    }
    Serial.println("");
  }
  
  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_SEL_VOLTAGE[8], seprated by comma (+/- COL_SEL_SPAN): ");
  }
  while (Serial.available() == 0) {}
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    COL_SEL_VOLTAGE[MY_COUNTER0] = Serial.parseFloat();
    if (COL_SEL_VOLTAGE[MY_COUNTER0] == (float)1024) return -1;
    //COL_SEL_VOLTAGE[MY_COUNTER0] = constrain(COL_SEL_VOLTAGE[MY_COUNTER0], 0, COL_DAC_SPAN);
    COL_SEL_VOLTAGE[MY_COUNTER0] = constrain(COL_SEL_VOLTAGE[MY_COUNTER0], -COL_SEL_SPAN, COL_SEL_SPAN);
  }

  if (Serial_println_ON == 1) {
    Serial.println("Enter TIA_A0(0-255): ");
  }

  while (Serial.available() == 0) {}
  TIA_A0 = Serial.parseInt();
  if (TIA_A0 == 1024) return -1;
  TIA_A0 = constrain(TIA_A0, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter TIA_A1(0-255): ");
  }

  while (Serial.available() == 0) {}
  TIA_A1 = Serial.parseInt();
  if (TIA_A1 == 1024) return -1;
  TIA_A1 = constrain(TIA_A1, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter COL_PULSE_WIDTH(0-1023): ");
  }

  while (Serial.available() == 0) {}
  COL_PULSE_WIDTH = Serial.parseInt();
  if (COL_PULSE_WIDTH == 1024) return -1;
  COL_PULSE_WIDTH = constrain(COL_PULSE_WIDTH, 0, 1023);

  if (Serial_println_ON == 1) {
    Serial.println("Enter SAMPLE_AND_HOLD(0-1023): ");
  }

  while (Serial.available() == 0) {}
  SAMPLE_AND_HOLD = Serial.parseInt();
  if (SAMPLE_AND_HOLD == 1024) return -1;
  SAMPLE_AND_HOLD = constrain(SAMPLE_AND_HOLD, 0, 1023);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ADC_CONVST(0-1023): ");
  }

  while (Serial.available() == 0) {}
  ADC_CONVST = Serial.parseInt();
  if (ADC_CONVST == 1024) return -1;
  ADC_CONVST = constrain(ADC_CONVST, 0, 1023);

  // Input configurations of column boards - end

  //select the column board, in demo it should be the frontmost column board - Begin****************************************
  //In Eric's code column board number should be 7 here.
  select_column_board(CB_NUMBER);
  //select column board - End****************************************

  //Setup MUX and TIA Gains on the Column Board - begin *************************

  //Set-up Column MUX Enable all to 0 at the first place ()********************************************
  //Unenable MUX_En to avoid simultaneous activate TIA and connect device
  send_to_latch_with_delay(0,0,1);//Enable selected MUX
  //Set-up Column MUX_A0 and MUX_A1********************************************
  send_to_latch_with_delay(1,COL_MUX_A0,1);
  send_to_latch_with_delay(2,COL_MUX_A1,1);
  //Set-up Column MUX Enable********************************************
  //Enable MUX_En after activation of TIA, which can avoid a very short negtive pulse applied to TEs.
  send_to_latch_with_delay(0,COL_MUX_EN,2);
  //Set-up TIA gain for all channels by configure TIA_A0 and TIA_A1********************************
  send_to_latch_with_delay(4,TIA_A0,1);
  send_to_latch_with_delay(5,TIA_A1,1);

  //Setup MUX and TIA Gains on Column Board - end********************************

  //Set Pulse Width on Column Board-begin*************************************************
  //Since we're trying to do a read, column board pulse width setting won't make any difference,
  //but we will set it to max to avoid switch noise in TIA.
  //Set Pulse Width on Column Board-end*************************************************
  setup_col_pulse_width(COL_PULSE_WIDTH, CB_NUMBER, COL_DAC_SPAN);
  //Set Sample and Hold Timing on Control Board-begin**************************************
  setup_col_sample_and_hold(SAMPLE_AND_HOLD, CB_NUMBER, COL_DAC_SPAN);
  //Set Sample and Hold Timing on Control Board-end**************************************

  //Set ADC ConvSt Timing on Column Board-begin**********************************************
  setup_col_adc_convst(ADC_CONVST, CB_NUMBER, COL_DAC_SPAN);

  //Set ADC ConvSt Timing on Column Board-end**********************************************

  //Set up DAC on the column board - begin ******************************************

  //set the column board DAC span - end
  setup_col_dac_span(COL_DAC_SPAN, CB_NUMBER);
  //setup the column board DAC voltage
  setup_col_dac_voltage(COL_DAC_VOLTAGE, CB_NUMBER, COL_DAC_SPAN);
  // Set up the column board select voltage on all 8 channels
  setup_col_sel_span(COL_SEL_SPAN, CB_NUMBER);
  setup_col_sel_voltage(COL_SEL_VOLTAGE, CB_NUMBER, COL_SEL_SPAN);
  //When doing a read, the pulse width set on the row boards is what matters.
  //Still Pointing at the same Column Board

  // Set up DAC on the column board - end *******************************************

  if (Serial_println_ON == 1) {
    Serial.println("Column board setup complete.");
  }
  return 0;
}

int configure_rows() {
  // input configurations for row boards - begin

  if (Serial_println_ON == 1) {
    Serial.println("Enter RB_NUMBER(0 to 7): ");
  }
  
  while (Serial.available() == 0) {}
  RB_NUMBER = Serial.parseInt();
  if (RB_NUMBER == 1024) return -1;
  RB_NUMBER = constrain(RB_NUMBER, 0, 7);

  // for rows 0 to 7
  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_PULSE_WIDTH(0 to 1023)");
  }
  while (Serial.available() == 0) {}
  ROW_PULSE_WIDTH = Serial.parseInt();
  if (ROW_PULSE_WIDTH == 1024) return -1;
  ROW_PULSE_WIDTH = constrain(ROW_PULSE_WIDTH, 0, 1023);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_DAC_SPAN(5 or 10)");
  }
  while (Serial.available() == 0) {}
  ROW_DAC_SPAN = Serial.parseInt();
  if (ROW_DAC_SPAN == 1024) return -1;
  if (ROW_DAC_SPAN > 7) {
    ROW_DAC_SPAN = 10;
  } else {
    ROW_DAC_SPAN = 5;
  }

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_EN_0to7 (0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_EN_0to7 = Serial.parseInt();
  if (ROW_MUX_EN_0to7 == 1024) return -1;
  ROW_MUX_EN_0to7 = constrain(ROW_MUX_EN_0to7, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A0_0to7 (0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_A0_0to7 = Serial.parseInt();
  if (ROW_MUX_A0_0to7 == 1024) return -1;
  ROW_MUX_A0_0to7 = constrain(ROW_MUX_A0_0to7, 0 , 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A1_0to7 (0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_A1_0to7 = Serial.parseInt();
  if (ROW_MUX_A1_0to7 == 1024) return -1;
  ROW_MUX_A1_0to7 = constrain(ROW_MUX_A1_0to7, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_DAC_VOLTAGE_0to7[8], seprated by comma (+/- ROW_DAC_SPAN): ");
  }
  while (Serial.available() == 0) {}
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = Serial.parseFloat();
    if (ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] == (float)1024) return -1;
    //ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_0to7[MY_COUNTER0], 0, ROW_DAC_SPAN);
    ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_0to7[MY_COUNTER0], -ROW_DAC_SPAN, ROW_DAC_SPAN);
  }

  // for rows 8 to 15.
  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_EN_8to15 (0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_EN_8to15 = Serial.parseInt();
  if (ROW_MUX_EN_8to15 == 1024) return -1;
  ROW_MUX_EN_8to15 = constrain(ROW_MUX_EN_8to15, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A0_8to15 (0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_A0_8to15 = Serial.parseInt();
  if (ROW_MUX_A0_8to15 == 1024) return -1;
  ROW_MUX_A0_8to15 = constrain(ROW_MUX_A0_8to15, 0 , 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A1_8to15(0-255): ");
  }
  while (Serial.available() == 0) {}
  ROW_MUX_A1_8to15 = Serial.parseInt();
  if (ROW_MUX_A1_8to15 == 1024) return -1;
  ROW_MUX_A1_8to15 = constrain(ROW_MUX_A1_8to15, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_DAC_VOLTAGE_8to15[8], seprated by comma (+/- ROW_DAC_SPAN): ");
  }
  while (Serial.available() == 0) {}
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = Serial.parseFloat();
    if (ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] == (float)1024) return -1;
    //ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0], 0, ROW_DAC_SPAN);
    ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0], -ROW_DAC_SPAN, ROW_DAC_SPAN);
  }
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
      Serial.print(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0]);
    }
    Serial.println("");
  }
  // input configuratiosn for row boards - end

  // Set-up all row boards - begin ***************************************************

  //Set up MUX on Row Board ********************************************************
  select_row_board(RB_NUMBER);

  //set up ROW_MUX_EN_0to7 all to 0 at first place.
  send_to_latch_with_delay(0,ROW_MUX_EN_0to7,1);//row MUX
  //set up row MUX address 0 to 7
  send_to_latch_with_delay(1,ROW_MUX_A0_0to7,1);//Row A0
  send_to_latch_with_delay(2,ROW_MUX_A1_0to7,1);//Row A1
  //set up ROW_MUX_EN_8to15 all to zero at first place (Not applied).

  send_to_latch_with_delay(4,ROW_MUX_EN_8to15,1);
  //set up row MUX address 8 to 15.
  send_to_latch_with_delay(5,ROW_MUX_A0_8to15,1);
  send_to_latch_with_delay(6,ROW_MUX_A1_8to15,1);
  //Set Up DAC on the Row Board-begin******************************************************
  //set up DAC span
  //Preset Pulse Width on Row Board *********************************************
  //            int ROW_PULSE_WIDTH;//some value between 0 and 1023, default is 200.
  setup_row_pulse_width(ROW_PULSE_WIDTH, RB_NUMBER, ROW_DAC_SPAN);
  setup_row_dac_span(ROW_DAC_SPAN, RB_NUMBER);

  //Set up DAC input signals - begin
  // Set up the row board DAC voltage on all 16 channels
  setup_row0to7_dac_voltage(ROW_DAC_VOLTAGE_0to7, RB_NUMBER, ROW_DAC_SPAN);
  setup_row8to15_dac_voltage(ROW_DAC_VOLTAGE_8to15, RB_NUMBER, ROW_DAC_SPAN);

  //We're still pointing at Row 8 Board (Frontmost Row Board)

  // Set-up all row boards - end ***************************************************

  if (Serial_println_ON == 1) {
    Serial.println("Row board setup complete.");
  }
  return 0;
}

int seperate_update() {
  if (Serial_println_ON == 1) {
    Serial.println("Select the part of configuration to update");
    Serial.println("1024: Return to top level;     1:CB_NUMBER;         2:COL_MUX_EN;        3:COL_MUX_A0;             4:COL_MUX_A1; ");
    Serial.println("5: COL_DAC_SPAN;      6:COL_DAC_VOLTAGE;   7:COL_SEL_VOLTAGE;   8:TIA_A0;                 9:TIA_A1; ");
    Serial.println("10:COL_PULSE_WIDTH;  11:SAMPLE_AND_HOLD;  12:ADC_CONVST;       13:RB_NUMBER;             14:ROW_PULSE_WIDTH; ");
    Serial.println("15:ROW_DAC_SPAN;     16:ROW_MUX_EN_0to7;  17:ROW_MUX_A0_0to7;  18:ROW_MUX_A1_0to7;       19:ROW_DAC_VOLTAGE_0to7; ");
    Serial.println("20:ROW_MUX_EN_8to15; 21:ROW_MUX_A0_8to15; 22:ROW_MUX_A1_8to15; 23:ROW_DAC_VOLTAGE_8to15" );
  }
  //Serial.println("go");
  while (Serial.available() == 0) {}
  int switch2 = Serial.parseInt();
  int speed_mode;
  switch (switch2) {
    //select 0;
    case 1024://select 0;
      return -1;
    case 1: //select 1
      if (Serial_println_ON == 1) {
        Serial.println("Enter the CB_NUMBER(0-7): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      CB_NUMBER = Serial.parseInt();
      if (CB_NUMBER == 1024) return -1;
      CB_NUMBER = constrain(CB_NUMBER, 0, 7);
      // excution:
      select_column_board(CB_NUMBER);
      break;

    case 2:
      //select 2
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_MUX_EN(0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      COL_MUX_EN = Serial.parseInt();
      if (COL_MUX_EN == 1024) return -1;
      COL_MUX_EN = constrain(COL_MUX_EN, 0, 255);
      if (Serial_println_ON == 1) {
        Serial.println(COL_MUX_EN);
      }
      // excution:

      select_column_board(CB_NUMBER);
      select_latch(0);//MUX Enable Lines
      delayMicroseconds(1);
      //Enable selected MUX
      PORTE = COL_MUX_EN;
      delayMicroseconds(1);
      pulse_the_latch();
      break;

    case 3:
      //select 3
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_MUX_A0(0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      COL_MUX_A0 = Serial.parseInt();
      if (COL_MUX_A0 == 1024) return -1;
      COL_MUX_A0 = constrain(COL_MUX_A0, 0 , 255);
      // excution:

      select_column_board(CB_NUMBER);
      select_latch(1);//MUX A0 Address Lines
      delayMicroseconds(1);
      PORTE = COL_MUX_A0;
      delayMicroseconds(1);
      pulse_the_latch();
      break;

    case 4:
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_MUX_A1(0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      COL_MUX_A1 = Serial.parseInt();
      if (COL_MUX_A1 == 1024) return -1;
      COL_MUX_A1 = constrain(COL_MUX_A1, 0, 255);
      // excution:
      select_column_board(CB_NUMBER);
      select_latch(2);//MUX A1 Address Lines
      delayMicroseconds(1);
      PORTE = COL_MUX_A1;
      delayMicroseconds(1);
      pulse_the_latch();
      break;

    case 5:
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_DAC_SPAN(5 or 10 for +/- 5 or +/- 10)");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      COL_DAC_SPAN = Serial.parseInt();
      if (COL_DAC_SPAN == 1024) return -1;
      if (COL_DAC_SPAN > 7) {
        COL_DAC_SPAN = 10;
      } else {
        COL_DAC_SPAN = 5;
      }

      setup_col_dac_span(COL_DAC_SPAN, CB_NUMBER);
      break;

    case 6:
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_DAC_VOLTAGE[8], seprated by comma (+/- COL_DAC_SPAN): ");
      }
      while (Serial.available() == 0) {}
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
        COL_DAC_VOLTAGE[MY_COUNTER0] = Serial.parseFloat();
        if (COL_DAC_VOLTAGE[MY_COUNTER0] == (float)1024) return -1;
        // COL_DAC_VOLTAGE[MY_COUNTER0] = constrain(COL_DAC_VOLTAGE[MY_COUNTER0], 0, COL_DAC_SPAN);
        COL_DAC_VOLTAGE[MY_COUNTER0] = constrain(COL_DAC_VOLTAGE[MY_COUNTER0], -COL_DAC_SPAN, COL_DAC_SPAN);
      }

      //excution:
      //Set up DAC on the column board - begin ******************************************
      setup_col_dac_span(COL_DAC_SPAN, CB_NUMBER);
      setup_col_dac_voltage(COL_DAC_VOLTAGE, CB_NUMBER, COL_DAC_SPAN);
      break;

    case 7:
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_SEL_VOLTAGE[8], seprated by comma (+/- COL_SEL_SPAN): ");
      }
      while (Serial.available() == 0) {}
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
        COL_SEL_VOLTAGE[MY_COUNTER0] = Serial.parseFloat();
        if (COL_SEL_VOLTAGE[MY_COUNTER0] == (float)1024) return -1;
        //COL_SEL_VOLTAGE[MY_COUNTER0] = constrain(COL_SEL_VOLTAGE[MY_COUNTER0], 0, COL_DAC_SPAN);
        COL_SEL_VOLTAGE[MY_COUNTER0] = constrain(COL_SEL_VOLTAGE[MY_COUNTER0], -COL_SEL_SPAN, COL_SEL_SPAN);
      }
      //excution:
      setup_col_sel_span(COL_SEL_SPAN, CB_NUMBER);
      setup_col_sel_voltage(COL_SEL_VOLTAGE, CB_NUMBER, COL_SEL_SPAN);
      break;

    case 8:
      if (Serial_println_ON == 1) {
        Serial.println("Enter TIA_A0(0-255): ");
      }
      while (Serial.available() == 0) {}
      TIA_A0 = Serial.parseInt();
      if (TIA_A0 == 1024) return -1;
      TIA_A0 = constrain(TIA_A0, 0, 255);

      select_column_board(CB_NUMBER);
	  send_to_latch_with_delay(4,TIA_A0,1);//TIA A0 Address
      break;
    case 9:
      if (Serial_println_ON == 1) {
        Serial.println("Enter TIA_A1(0-255): ");
      }
      while (Serial.available() == 0) {}
      TIA_A1 = Serial.parseInt();
      if (TIA_A1 == 1024) return -1;
      TIA_A1 = constrain(TIA_A1, 0, 255);

      select_column_board(CB_NUMBER);
	  send_to_latch_with_delay(5,TIA_A1,1);//TIA A1 Address
      break;
    case 10:
      if (Serial_println_ON == 1) {
        Serial.println("Enter COL_PULSE_WIDTH(0-1023): ");
      }
      while (Serial.available() == 0) {}
      COL_PULSE_WIDTH = Serial.parseInt();
      if (COL_PULSE_WIDTH == 1024) return -1;
      COL_PULSE_WIDTH = constrain(COL_PULSE_WIDTH, 0, 1023);

      //Set Pulse Width on Column Board-begin*************************************************
      setup_col_pulse_width(COL_PULSE_WIDTH, CB_NUMBER, COL_DAC_SPAN);
      //Set Pulse Width on Column Board-end*************************************************
      break;
    case 11:
      if (Serial_println_ON == 1) {
        Serial.println("Enter SAMPLE_AND_HOLD(0-1023): ");
      }
      while (Serial.available() == 0) {}
      SAMPLE_AND_HOLD = Serial.parseInt();
      if (SAMPLE_AND_HOLD == 1024) return -1;
      SAMPLE_AND_HOLD = constrain(SAMPLE_AND_HOLD, 0, 1023);
      //Set Sample and Hold Timing on Control Board-begin**************************************
      setup_col_sample_and_hold(SAMPLE_AND_HOLD, CB_NUMBER, COL_DAC_SPAN);
      //Set Sample and Hold Timing on Control Board-end**************************************
      break;
    case 12:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ADC_CONVST(0-1023): ");
      }
      while (Serial.available() == 0) {}
      ADC_CONVST = Serial.parseInt();
      if (ADC_CONVST == 1024) return -1;
      ADC_CONVST = constrain(ADC_CONVST, 0, 1023);

      //Set ADC ConvSt Timing on Column Board-begin**********************************************
      setup_col_adc_convst(ADC_CONVST, CB_NUMBER, COL_DAC_SPAN);
      //Set ADC ConvSt Timing on Column Board-end**********************************************
      break;
    case 13:
      // Input configurations of column boards - end
      if (Serial_println_ON == 1) {
        Serial.println("Enter the RB_NUMBER(0 to 7): ");
      }
      while (Serial.available() == 0) {}
      RB_NUMBER = Serial.parseInt();
      if (RB_NUMBER == 1024) return -1;
      RB_NUMBER = constrain(RB_NUMBER, 0, 7);

      select_row_board(RB_NUMBER);
      break;
    // for rows 0 to 7
    case 14:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_PULSE_WIDTH(0 to 1023)");
      }
      while (Serial.available() == 0) {}
      ROW_PULSE_WIDTH = Serial.parseInt();
      if (ROW_PULSE_WIDTH == 1024) return -1;
      ROW_PULSE_WIDTH = constrain(ROW_PULSE_WIDTH, 0, 1023);

      setup_row_pulse_width(ROW_PULSE_WIDTH, RB_NUMBER, ROW_DAC_SPAN);
      break;

    case 15:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_DAC_SPAN(5 or 10)");
      }
      while (Serial.available() == 0) {}
      ROW_DAC_SPAN = Serial.parseInt();
      if (ROW_DAC_SPAN == 1024) return -1;
      if (ROW_DAC_SPAN > 7) {
        ROW_DAC_SPAN = 10;
      } else {
        ROW_DAC_SPAN = 5;
      }

      //setup_row_pulse_width(ROW_PULSE_WIDTH, RB_NUMBER, ROW_DAC_SPAN);
      setup_row_dac_span(ROW_DAC_SPAN, RB_NUMBER);
      //setup_row0to7_dac_voltage(ROW_DAC_VOLTAGE_0to7, RB_NUMBER, ROW_DAC_SPAN);
      //setup_row8to15_dac_voltage(ROW_DAC_VOLTAGE_8to15, RB_NUMBER, ROW_DAC_SPAN);

      //setup_row_pulse_width(ROW_PULSE_WIDTH, RB_NUMBER, ROW_DAC_SPAN);
      break;

    case 16:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_EN_0to7 (0-255): ");
      }
      while (Serial.available() == 0) {}
      ROW_MUX_EN_0to7 = Serial.parseInt();
      if (ROW_MUX_EN_0to7 == 1024) return -1;
      ROW_MUX_EN_0to7 = constrain(ROW_MUX_EN_0to7, 0, 255);

      select_row_board(RB_NUMBER);
      //set up ROW_MUX_EN_0to7
	  send_to_latch_with_delay(0,ROW_MUX_EN_0to7,1);//Row Enable Lines
      break;

    case 17:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_A0_0to7 (0-255): ");
      }
      while (Serial.available() == 0) {}
      ROW_MUX_A0_0to7 = Serial.parseInt();
      if (ROW_MUX_A0_0to7 == 1024) return -1;
      ROW_MUX_A0_0to7 = constrain(ROW_MUX_A0_0to7, 0 , 255);

      select_row_board(RB_NUMBER);
	  send_to_latch_with_delay(1,ROW_MUX_A0_0to7,1);//Row A0 Address Lines

      break;

    case 18:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_A1_0to7 (0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      ROW_MUX_A1_0to7 = Serial.parseInt();
      if (ROW_MUX_A1_0to7 == 1024) return -1;
      ROW_MUX_A1_0to7 = constrain(ROW_MUX_A1_0to7, 0, 255);

      select_row_board(RB_NUMBER);
	  send_to_latch_with_delay(2,ROW_MUX_A1_0to7,1);//Row A1 Address Lines
      break;

    case 19:
      //            int MY_COUNTER0 = 0;
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_DAC_VOLTAGE_0to7[8], seprated by comma (+/- ROW_DAC_SPAN): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
        ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = Serial.parseFloat();
        if (ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] == (float)1024) return -1;
        //ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_0to7[MY_COUNTER0], 0, ROW_DAC_SPAN);
        ROW_DAC_VOLTAGE_0to7[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_0to7[MY_COUNTER0], -ROW_DAC_SPAN, ROW_DAC_SPAN);
      }
      setup_row_dac_span(ROW_DAC_SPAN, RB_NUMBER);
      setup_row0to7_dac_voltage(ROW_DAC_VOLTAGE_0to7, RB_NUMBER, ROW_DAC_SPAN);
      break;

    // for rows 8 to 15.
    case 20:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_EN_8to15 (0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      ROW_MUX_EN_8to15 = Serial.parseInt();
      if (ROW_MUX_EN_8to15 == 1024) return -1;
      ROW_MUX_EN_8to15 = constrain(ROW_MUX_EN_8to15, 0, 255);

      //set up ROW_MUX_EN_8to15
      select_row_board(RB_NUMBER);
	  send_to_latch_with_delay(4,ROW_MUX_EN_8to15,1);//Row Enable Lines
      break;
    case 21:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_A0_8to15 (0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      ROW_MUX_A0_8to15 = Serial.parseInt();
      if (ROW_MUX_A0_8to15 == 1024) return -1;
      ROW_MUX_A0_8to15 = constrain(ROW_MUX_A0_8to15, 0 , 255);

      //set up row MUX address 8 to 15.
      select_row_board(RB_NUMBER);
	  send_to_latch_with_delay(5,ROW_MUX_A0_8to15,1);//Row A0 Address Lines
      break;
    case 22:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_MUX_A1_8to15(0-255): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      ROW_MUX_A1_8to15 = Serial.parseInt();
      if (ROW_MUX_A1_8to15 == 1024) return -1;
      ROW_MUX_A1_8to15 = constrain(ROW_MUX_A1_8to15, 0, 255);

      select_row_board(RB_NUMBER);
	  send_to_latch_with_delay(6,ROW_MUX_A1_8to15,1);//Row A1 Address Lines
      break;
    case 23:
      if (Serial_println_ON == 1) {
        Serial.println("Enter ROW_DAC_VOLTAGE_8to15[8], seprated by comma (+/- ROW_DAC_SPAN): ");
      }
      //Serial.println("go");
      while (Serial.available() == 0) {}
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
        ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = Serial.parseFloat();
        if (ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] == (float)1024) return -1;
        // ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0], 0, ROW_DAC_SPAN);
        ROW_DAC_VOLTAGE_8to15[MY_COUNTER0] = constrain(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0], -ROW_DAC_SPAN, ROW_DAC_SPAN);
      }
      setup_row_dac_span(ROW_DAC_SPAN, RB_NUMBER);
      setup_row8to15_dac_voltage(ROW_DAC_VOLTAGE_8to15, RB_NUMBER, ROW_DAC_SPAN);
      break;
    case 24:
      if (Serial_println_ON == 1) Serial.print("ADC Return: ");
      select_column_board(CB_NUMBER);
      select_latch(6); //Preselect latch with ADC control lines
      PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
      pulse_the_latch();
      //delayMicroseconds(1);
      //digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
      //delayMicroseconds(5);
      //digitalWrite(MPU_PULSE, LOW);
      PORTE = 5;//Lower ADC Not Chip Select
      digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
      for (int i = 0; i < 8; i++)
      {
        PORTE = B00000101;//5;//Raise ADC_NOT_READ
        PORTE = B00000001;//1;//Lower ADC_NOT_READ
        delayMicroseconds(1);
        ADC_READ_VALUE[i] = PORTD;
      }
      PORTE = B00000101;//5;//Raise ADC_NOT_READ
      PORTE = B00000111;//7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
      digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches
      for (int i = 0; i < 7; i++)
      {
        Serial.print(ADC_READ_VALUE[i]);
        Serial.print(",");
      }
      Serial.println(ADC_READ_VALUE[7]);

      break;
    case 25:// TODO ???
      if (Serial_println_ON == 1) Serial.println("Enter Fast(0)or Slow(1)Way:");
      while (Serial.available() == 0) {}
      speed_mode = Serial.parseInt();
      if (speed_mode == 1024) return -1;
      speed_mode = constrain(speed_mode, 0, 1);
      select_row_board(CB_NUMBER);
      select_latch(3);//Row A1 Address Lines
      if (speed_mode == 1) PORTE = 43; // 0010 1011.
      else PORTE = 42;
      pulse_the_latch();
      break;
    case 26:// TODO ???
      if (Serial_println_ON == 1) Serial.println("Enter Fast(0)or Slow(1)Way:");
      while (Serial.available() == 0) {}
      speed_mode = Serial.parseInt();
      if (speed_mode == 1024) return -1;
      speed_mode = constrain(speed_mode, 0, 1);
      select_row_board(RB_NUMBER);
      select_latch(3);//Row A1 Address Lines
      if (speed_mode == 1) PORTE = 43; // 0010 1011.
      else PORTE = 42;
      pulse_the_latch();
      break;
    case 27: // TODO ???
      if (Serial_println_ON == 1) Serial.println("Enter Fast(0)or Slow(1)Way:");
      while (Serial.available() == 0) {}
      speed_mode = Serial.parseInt();
      digitalWrite(SLOW_PULSE, HIGH);//Send Pulse That Starts the Entire Process
      delayMicroseconds(speed_mode);
      digitalWrite(SLOW_PULSE, LOW);
      break;
  }
  return 0;
}

int fast_read() {
  if (Serial_println_ON == 1) {
    Serial.println("Enter number of column board (0 to 7) to read from: ");
  }
  //Serial.println("go");
  while (Serial.available() == 0) {}
  CB_NUMBER = Serial.parseInt();
  if (CB_NUMBER == 1024) return -1;
  CB_NUMBER = constrain(CB_NUMBER , 0, 7);

  //select_row_board(RB_NUMBER);

  select_column_board(CB_NUMBER);
  //Preselect latch with ADC control lines
  select_latch(6);
  PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
  pulse_the_latch();

  delayMicroseconds(1);
  digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
  delayMicroseconds(5);
  digitalWrite(MPU_PULSE, LOW);

  //select_row_board(RB_NUMBER);
  //row_safe_start();

  //Probably should look at "END OF CONVERT" signal now, but it happens so fast
  //that I'm just putting in a small delay before doing reads.
  select_column_board(CB_NUMBER);
  //delayMicroseconds(1);
  PORTE = B00000101;//5;//Lower ADC Not Chip Select
  //delayMicroseconds(1);
  //Do 8 Reads to Return ADC Values
  digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    PORTE = 5;//Raise ADC_NOT_READ
    PORTE = 1;//Lower ADC_NOT_READ
    delayMicroseconds(1);
    ADC_READ_VALUE[MY_COUNTER0] = PORTD;
  }
  PORTE = 5;//Raise ADC_NOT_READ
  PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches

  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    Serial.print(ADC_READ_VALUE[MY_COUNTER0]);
    Serial.print(",");
  }
  Serial.println(ADC_READ_VALUE[7]);

  if (Serial_println_ON == 1) {
	  float V_CONVERT[8];
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
      V_CONVERT[MY_COUNTER0] = (ADC_READ_VALUE[MY_COUNTER0] - (float)32768) * (float)10 / (float)65536;
      if (MY_COUNTER0 < 7) {
        Serial.print(V_CONVERT[MY_COUNTER0]);
        Serial.print(",");
      } else {
        Serial.println(V_CONVERT[MY_COUNTER0]);
      }
    }
  }
  //if(Serial_println_ON == 1) {
  Serial.println("ADC read complete");
  //}
  return 0;
}

void display_configuration() {
  Serial.println("Column board configurations:");
  Serial.println(CB_NUMBER);
  Serial.println(COL_MUX_EN);
  Serial.println(COL_MUX_A0);
  Serial.println(COL_MUX_A1);
  Serial.println(COL_DAC_SPAN);
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    Serial.print(COL_DAC_VOLTAGE[MY_COUNTER0]);
    Serial.print(", ");
  }
  Serial.println(COL_DAC_VOLTAGE[7]);
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    Serial.print(COL_SEL_VOLTAGE[MY_COUNTER0]);
    Serial.print(", ");
  }
  Serial.println(COL_SEL_VOLTAGE[7]);
  Serial.println(TIA_A0);
  Serial.println(TIA_A1);
  Serial.println(COL_PULSE_WIDTH);
  Serial.println(SAMPLE_AND_HOLD);
  Serial.println(ADC_CONVST);
  Serial.println("Row board configurations:");
  Serial.println(RB_NUMBER);
  Serial.println(ROW_PULSE_WIDTH);
  Serial.println(ROW_DAC_SPAN);
  Serial.println(ROW_MUX_EN_0to7);
  Serial.println(ROW_MUX_A0_0to7);
  Serial.println(ROW_MUX_A1_0to7);
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    Serial.print(ROW_DAC_VOLTAGE_0to7[MY_COUNTER0]);
    Serial.print(", ");
  }
  Serial.println(ROW_DAC_VOLTAGE_0to7[7]);
  Serial.println(ROW_MUX_EN_8to15);
  Serial.println(ROW_MUX_A0_8to15);
  Serial.println(ROW_MUX_A1_8to15);
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    Serial.print(ROW_DAC_VOLTAGE_8to15[MY_COUNTER0]);
    Serial.print(", ");
  }
  Serial.println(ROW_DAC_VOLTAGE_8to15[7]);

}

int mpu_read() {//slow read, use mpu pulse, delayMicroseconds(SLOW_PULSE_WIDTH - 10);
	//only used in legacy code, not going to maintain
  if (Serial_println_ON == 1) {
    Serial.println("Enter number of row board (0 to 7) to write to crossbar: ");
  }
  while (Serial.available() == 0) {}
  RB_NUMBER = Serial.parseInt();
  if (RB_NUMBER == 1024) return -1;
  RB_NUMBER = constrain(RB_NUMBER , 0, 7);

  if (Serial_println_ON == 1) {
    Serial.println("Enter number of column board (0 to 7) to read from: ");
  }
  while (Serial.available() == 0) {}
  CB_NUMBER = Serial.parseInt();
  if (CB_NUMBER == 1024) return -1;
  CB_NUMBER = constrain(CB_NUMBER , 0, 7);

  if (Serial_println_ON == 1) {
    Serial.println("Enter the slow pulse width (us): ");
  }
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  if (SLOW_PULSE_WIDTH == 1024) return -1;
  SLOW_PULSE_WIDTH = constrain(SLOW_PULSE_WIDTH, 10, 1000000);

  //CB_NUMBER = constrain(CB_NUMBER ,0,7);
  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A0_0to7_SLOW (0-255): ");
  }
  while (Serial.available() == 0) {}
  int ROW_MUX_A0_0to7_SLOW = Serial.parseInt();
  if (ROW_MUX_A0_0to7_SLOW == 1024) return -1;
  ROW_MUX_A0_0to7_SLOW = constrain(ROW_MUX_A0_0to7_SLOW, 0 , 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A1_0to7_SLOW (0-255): ");
  }
  while (Serial.available() == 0) {}
  int ROW_MUX_A1_0to7_SLOW = Serial.parseInt();
  if (ROW_MUX_A1_0to7_SLOW == 1024) return -1;
  ROW_MUX_A1_0to7_SLOW = constrain(ROW_MUX_A1_0to7_SLOW, 0, 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A0_8to15_SLOW (0-255): ");
  }
  while (Serial.available() == 0) {}
  int ROW_MUX_A0_8to15_SLOW = Serial.parseInt();
  if (ROW_MUX_A0_8to15_SLOW == 1024) return -1;
  ROW_MUX_A0_8to15_SLOW = constrain(ROW_MUX_A0_8to15_SLOW, 0 , 255);

  if (Serial_println_ON == 1) {
    Serial.println("Enter ROW_MUX_A1_8to15_SLOW(0-255): ");
  }
  while (Serial.available() == 0) {}
  int ROW_MUX_A1_8to15_SLOW = Serial.parseInt();
  if (ROW_MUX_A1_8to15_SLOW == 1024) return -1;
  ROW_MUX_A1_8to15_SLOW = constrain(ROW_MUX_A1_8to15_SLOW, 0, 255);

  select_column_board(CB_NUMBER);
  //Preselect latch with ADC control lines
  //Set up ADC Control Latch for main loop
  send_to_latch(6,B00000111);//7, Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
  select_row_board(RB_NUMBER);
  send_to_latch(2,ROW_MUX_A1_0to7_SLOW);
  send_to_latch(6,ROW_MUX_A1_8to15_SLOW);

  if (ROW_MUX_A0_0to7_SLOW == 0) {//OLDTODO why?? If not 0~7, send mpu pulse to latch5(8~15)? then what if both are not 0?
    select_latch(1);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_0to7_SLOW;
    pulse_the_latch();
    select_latch(5);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_8to15_SLOW;
  } else {
    select_latch(5);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_8to15_SLOW;
    pulse_the_latch();
    select_latch(1);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_0to7_SLOW;
  }

  digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
  delayMicroseconds(SLOW_PULSE_WIDTH - 10);
  digitalWrite(MPU_PULSE, HIGH);
  delayMicroseconds(5);
  digitalWrite(MPU_PULSE, LOW);

  if (ROW_MUX_A0_0to7_SLOW == 0) {
    PORTE = ROW_MUX_A0_8to15;
    select_latch(1);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_0to7;
    pulse_the_latch();

  } else {
    PORTE = ROW_MUX_A0_0to7;
    select_latch(5);//Row A0 Address Lines
    PORTE = ROW_MUX_A0_8to15;
    pulse_the_latch();

  }
  send_to_latch(2,ROW_MUX_A1_0to7);
  send_to_latch(6,ROW_MUX_A1_8to15);
  //Probably should look at "END OF CONVERT" signal now, but it happens so fast
  //that I'm just putting in a small delay before doing reads.
  select_column_board(CB_NUMBER);
  delayMicroseconds(1);
  select_latch(6);
  PORTE = 5;//Lower ADC Chip Select
  //Do 8 Reads to Return ADC Values
  delayMicroseconds(1);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    PORTE = 1;//Lower ADC_NOT_READ
    ADC_READ_VALUE[MY_COUNTER0] = PORTD;
    PORTE = 5;//Raise ADC_NOT_READ
    delayMicroseconds(1);
  }
  PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
  delayMicroseconds(1);

  float TEMP1 = ADC_READ_VALUE[0];//TODO why rotate here?

  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
    ADC_READ_VALUE[MY_COUNTER0] = ADC_READ_VALUE[MY_COUNTER0 + 1];
  }
  ADC_READ_VALUE[7] = TEMP1;

  for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
    Serial.print(ADC_READ_VALUE[MY_COUNTER0]);
    Serial.print(", ");
  }
  Serial.println("");

  if (Serial_println_ON == 1) {
    Serial.println("ADC read complete.");
  }
  return 0;
}

void batch_reset() {//reset a subarray from the lower corner
  // Read in batch set configurations - begin.
  if (Serial_println_ON == 1) {
    Serial.println("Batch set command 1: number of column boards(1~8), number of row boards(1~8), row pulse width(0~1023), COL_DAC_BATCH_SPAN, ROW_DAC_BATCH_SPAN, upper(0) or bottom(1)");
  }
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  ROW_BATCH_PULSE_WIDTH = Serial.parseInt();
  while (Serial.available() == 0) {}
  COL_DAC_BATCH_SPAN = Serial.parseInt();
  while (Serial.available() == 0) {}
  ROW_DAC_BATCH_SPAN = Serial.parseInt();
  // while (Serial.available() == 0) {}
  // bottom_flag = Serial.parseInt();

  int TOTAL_COLS = 8 * NUM_COL_BOARDS;
  int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;
  // if (TOTAL_ROWS > 64) {
  //   TOTAL_ROWS = 64;
  // }
  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",");
    Serial.print("ROW_BATCH_PULSE_WIDTH: ");
    Serial.print(ROW_BATCH_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    Serial.print(",");
    Serial.print("ROW_DAC_BATCH_SPAN: ");
    Serial.print(ROW_DAC_BATCH_SPAN);
    // Serial.print(",");
    // Serial.print("bottom_flag: ");
    // Serial.println(bottom_flag);
  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch set command 2: V_BATCH_SET for row boards. Array size is previously defined by number of column and row boards.");
  }
  // Read voltage signal row by row;

  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      V_BATCH_SET[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }
  if (Serial_println_ON == 1) {
    Serial.println("Batch set command 3: V_BATCH_SET_GATE for column boards. Array size is previously defined by number of column and row boards.");
  }

  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  // Step 1:
  // loop setup all existing column boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2++) {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in reading mode: 0, 0, 255
    // Here the MUX are all disabled, and the all columns are grounded.
	//floating all cols
	send_to_latch(0,0);//MUX Enable Lines
	send_to_latch(1,0);//MUX A0 Address Lines
	send_to_latch(2,0);//MUX A1 Address Lines

    //No need to setup TIAs since it is in set mode********************************
    //setup_col_pulse_width(1024, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    //setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN); // set all column DAC device voltages to 0.
    //COL_SEL_BATCH_VOLTAGE_ZERO is always 0.
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);//TODO WARNING: global default value used
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN); // set all column DAC selector votlage to 0.

  }

  if (Serial_println_ON == 1) {
    Serial.println("Loop setup all existing column boards complete.");
  }

  // Step 2:
  // loop setup all existing row boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in fast_pulse mode.
    //set up ROW_MUX_EN_0to7
	send_to_latch(0,0);//Row Enable Lines
	send_to_latch(1,0);//Row A0 Address Lines
	send_to_latch(2,0);//Row A1 Address Lines
	//set up ROW_MUX_EN_8to15
	send_to_latch(4,0);//Row Enable Lines
	send_to_latch(5,0);//Row A0 Address Lines
	send_to_latch(6,0);//Row A1 Address Lines
    setup_row_pulse_width(ROW_BATCH_PULSE_WIDTH, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    //setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  if (Serial_println_ON == 1) {
    Serial.println("Loop setup all existing row boards complete.");
  }
  // setup all column boards and row boards before batch set - end.

  // Begin batch set:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 <  TOTAL_ROWS; MY_COUNTER2++) { // column-wise loop
    for (int MY_COUNTER1 = 0; MY_COUNTER1 <  TOTAL_COLS; MY_COUNTER1++) { // row-wise loop

      int current_col = MY_COUNTER1;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

      int current_row = MY_COUNTER2;
	  int current_row_in_rb = current_row % 16;
	  int current_rb_number = current_row / 16;
      // if (bottom_flag > 0) {
		  // current_rb_number += 4;;
      // }

	  if (V_BATCH_SET[current_row][current_col] > 0) {
		  // set row dac TODO:can we change the order? move to next
		  select_row_board(current_rb_number);
		  setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
		  setup_row_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
		  //ground the column, set column gate
		  select_column_board(current_cb_number);
		  send_to_latch(0, PORTE_conversion_for_single_channel(current_col_in_cb));//MUX EN Address Lines

		  setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
		  setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
		  //delayMicroseconds(10);//TODO WTF 10 us
		  //enable the row to fast
		  select_row_board(current_rb_number);
		  if (current_row_in_rb < 8) {
			  send_to_latch(1, PORTE_conversion_for_single_channel(current_row_in_rb));//A0
			  send_to_latch(2, PORTE_conversion_for_single_channel(current_row_in_rb));//A1
			  send_to_latch(0, PORTE_conversion_for_single_channel(current_row_in_rb));//MUX_EN
		  }
		  else {
			  send_to_latch(5, PORTE_conversion_for_single_channel(current_row_in_rb - 8));//A0
			  send_to_latch(6, PORTE_conversion_for_single_channel(current_row_in_rb - 8));//A1
			  send_to_latch(4, PORTE_conversion_for_single_channel(current_row_in_rb - 8));//MUX_EN
		  }
		  //SEND the pulse
		  digitalWrite(MPU_PULSE, HIGH);
		  delayMicroseconds(10);
		  digitalWrite(MPU_PULSE, LOW);

		  //reset column gate voltage to 0. TODO exchangeable with following row reset op? Yes
		  select_column_board(current_cb_number);
		  send_to_latch(0, 0);//MUX_EN
		  setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
		  setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
		  //disable the rows TODO: actually disabled all the row on the rb?
		  select_row_board(current_rb_number);
		  if (current_row_in_rb < 8) {
			  send_to_latch(1, 0);//A0
			  send_to_latch(2, 0);//A1
			  send_to_latch(0, 0);//MUX_EN
		  }
		  else {
			  send_to_latch(5, 0);//A0
			  send_to_latch(6, 0);//A1
			  send_to_latch(4, 0);//MUX_EN
		  }
		  //reset row voltage to 0
		  setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
		  setup_row_dac_voltage_oneline(0, current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
	  }
	}
  }

  // Use memristor model if needed

  if (MODEL_ON == 1) {}

  safeStartAllRowsAndColumns();
  // Print task complete message.
  Serial.println("cmd11 batch reset fast complete");

}

void batch_set() {
	//mostly dupe from batch_set_slow();
  // Read in batch reset configurations - begin.
  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command 1: number of column boards(1~8), number of row boards(1~8), column pulse width(0~1023), COL_DAC_BATCH_SPAN(5 or 10), bottom_flag");
  }
  while (Serial.available() == 0) {}
  int NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  int NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  int COL_BATCH_PULSE_WIDTH = Serial.parseInt();
  while (Serial.available() == 0) {}
  int COL_DAC_BATCH_SPAN = Serial.parseInt();
  // while (Serial.available() == 0) {}
  // int bottom_flag = Serial.parseInt();

  int TOTAL_COLS = 8 * NUM_COL_BOARDS;
  int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;
  // if (TOTAL_ROWS > 64) {
  //   TOTAL_ROWS = 64;
  // }
  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",");
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("COL_BATCH_PULSE_WIDTH: ");
    Serial.print(COL_BATCH_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    // Serial.print(",");
    // Serial.print("bottom_flag: ");
    // Serial.print(bottom_flag);
    Serial.println("");
  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command 2: V_BATCH_RESET for column boards. Array size is previously defined by number of column and row boards.");
  }
  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      // To save memory, use V_BATCH_SET array for both set and reset operations.
      V_BATCH_SET[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command 3: V_BATCH_RESET_GATE for column boards. Array size is previously defined by number of column and row boards.");
  }

  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      // To save memory, use V_BATCH_SET_GATE array for both set and reset operations.
      V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  // Read in Batch reset configurations - complete.

  // Step 1:
  // loop setup all existing column boards:
  // Column MUX configuration table:
  // States:     Float Float Float Float Ground Slow_pulse Read Fast_pulse
  // COL_MUX_A0:   0     1     0     1      0       1        0      1
  // COL_MUX_A1:   0     0     1     1      0       0        1      1
  // COL_MUX_EN:   0     0     0     0      1       1        1      1
  if (Serial_println_ON == 1) {
    Serial.println("Start loop setup all existing column boards");
  }
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2 = MY_COUNTER2 + 1) {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
	// Floating all columns while they are all in fast pulse mode: 0, 255, 255
	//floating all cols
	send_to_latch(0, 0);//Column MUX_EN
	send_to_latch(1, 0);//A0
	send_to_latch(2, 0);//A1

    //No need to setup TIAs since it is in reset mode********************************
	setup_col_pulse_width(COL_BATCH_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    //setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);

  }
  if (Serial_println_ON == 1) {
    Serial.println("Complete Loop setup all existing column boards");
  }
  // Step 2:
  // loop setup all existing row boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in grounding mode.
    //set up ROW_MUX_EN_0to7
    send_to_latch(0,0);//Row MUX_EN
    send_to_latch(1,0);//A0
    send_to_latch(2,0);//A1

    //set up ROW_MUX_EN_8to15
    send_to_latch(4,0);//Row MUX_EN
    send_to_latch(5,0);//A0
    send_to_latch(6,0);//A1
	//TODO maybe not need to config this cause it's set operation? Yes
    setup_row_pulse_width(1024, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    //setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  if (Serial_println_ON == 1) {
    Serial.println("Loop setup all existing row boards");
  }
  // setup all column boards and row boards before batch reset - end.

  // Begin batch reset:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 <  TOTAL_ROWS; MY_COUNTER2++) { // column-wise loop
    for (int MY_COUNTER1 = 0; MY_COUNTER1 <  TOTAL_COLS; MY_COUNTER1++) { // row-wise loop

      int current_row = MY_COUNTER2;
      int current_row_in_rb = current_row % 16;
	  int current_rb_number = current_row / 16;
      // if (bottom_flag > 0) {// bug: was 1
      //   current_rb_number +=4;
      // }

      int current_col = MY_COUNTER1;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

			if (V_BATCH_SET[current_row][current_col] != 0)
			{
        //enable the row
        // setup the gate voltage for the selected column.
        select_row_board(current_rb_number);
        if (current_row_in_rb < 8) {
          send_to_latch(0, PORTE_conversion_for_single_channel(current_row_in_rb));//Row MUX_EN
        } else {
          send_to_latch(4, PORTE_conversion_for_single_channel(current_row_in_rb-8));//Row MUX_EN
        }
		//set the column voltage and gate
        select_column_board(current_cb_number);
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
        delayMicroseconds(10);
        //enable the column
        send_to_latch(1,PORTE_conversion_for_single_channel(current_col_in_cb));//A0
        send_to_latch(2,PORTE_conversion_for_single_channel(current_col_in_cb));//A1
        send_to_latch(0,PORTE_conversion_for_single_channel(current_col_in_cb));//MUX_EN
		//SEND the pulse
        digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
        delayMicroseconds(5);
        digitalWrite(MPU_PULSE, LOW);

        // close the selector by reseting the column voltage and gate to 0 at the first place.
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);
        // disable the selected column TODO: actually disable all the columns
		send_to_latch(1,0);//A0
		send_to_latch(2,0);//A1
		send_to_latch(0,0);//EN
		//disable the rows
        select_row_board(current_rb_number);
		send_to_latch(0,0);//EN
		send_to_latch(4,0);//EN
      }
    }
  }

  // Use memristor model if needed

  if (MODEL_ON == 1) {}

  safeStartAllRowsAndColumns();

	Serial.println("cmd12 Batch set fast complete");
}
//There is a capital version that does exactly grounding.
void ground_all_rows_cols() {
	safeStartAllRowsAndColumns();//do more than that.

  Serial.println("Grounding complete");

}

void mpu_write() {//slow write to a 16x8 board
	//only used in lagacy code, not going to maintain
  // Read in MPU write configurations - begin.
  if (Serial_println_ON == 1) {
    Serial.println("Slow write command: RB_NUMBER, CB_NUMBER, ROW_MUX_EN_0to7_SLOW, ROW_MUX_EN_8to15_SLOW, COL_MUX_EN_SLOW, SLOW_PULSE_WIDTH, V_POLARITY");
  }
  while (Serial.available() == 0) {}
  RB_NUMBER = Serial.parseInt();
  while (Serial.available() == 0) {}
  CB_NUMBER = Serial.parseInt();
  while (Serial.available() == 0) {}
  int ROW_MUX_EN_0to7_SLOW = Serial.parseInt();
  while (Serial.available() == 0) {}
  int ROW_MUX_EN_8to15_SLOW = Serial.parseInt();
  while (Serial.available() == 0) {}
  int COL_MUX_EN_SLOW = Serial.parseInt();
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  SLOW_PULSE_WIDTH = constrain(SLOW_PULSE_WIDTH, 20, 1000000);
  while (Serial.available() == 0) {}
  int V_POLARITY = Serial.parseInt();
  // Read in MPU write configurations - end.
  // slow pulse MUX config:EN-1,A0-1,A1-0.
  int ROW_MUX_A0_0to7_SLOW = ROW_MUX_EN_0to7_SLOW;
  //int ROW_MUX_A1_0to7_SLOW = 0;
  int ROW_MUX_A0_8to15_SLOW = ROW_MUX_EN_8to15_SLOW;
  //int ROW_MUX_A1_8to15_SLOW = 0;
  int COL_MUX_A0_SLOW = COL_MUX_EN_SLOW;
  //int COL_MUX_A1_SLOW = 0;//unused variable

  // MPU write start
  // The positive voltage is from the column board
  if (V_POLARITY == 0) {
	  //enable the rows
    select_row_board(RB_NUMBER);
	send_to_latch(0,ROW_MUX_EN_0to7_SLOW);//Enables 1-8 On
	send_to_latch(4,ROW_MUX_EN_8to15_SLOW);//Enables 9-15 On

    // Set the selected column lines into slow pulse mode, other column lines are in floating mode.
    select_column_board(CB_NUMBER);
	send_to_latch(1,COL_MUX_A0_SLOW);//Col A0 Address Lines
	send_to_latch(0,COL_MUX_EN_SLOW);// Enable selected column, the voltage signal is out.
    // just delay, there is no need to MPU pulse. Slow
    delayMicroseconds(SLOW_PULSE_WIDTH - 10);

    // put Col A0 to 0 to ground the columns.
	send_to_latch(1,0);//A0
    // restore columns to previous COL_MUX_EN condition.TODO: global variable
	send_to_latch(0,COL_MUX_EN);

    // restore row boards to previous states.
    select_row_board(RB_NUMBER);
	send_to_latch(0,ROW_MUX_EN_0to7);//Enables 1-8 On
	send_to_latch(4,ROW_MUX_EN_8to15); //Enables 9-15 On

  } else {// The positive voltage is from the row board
    //enable the columns
    select_column_board(CB_NUMBER);
	send_to_latch(0,COL_MUX_EN_SLOW);
	//set the selected rows to slow pulse mode
    select_row_board(RB_NUMBER);
	send_to_latch(1,ROW_MUX_A0_0to7_SLOW);
	send_to_latch(5,ROW_MUX_A0_8to15_SLOW);
	send_to_latch(0,ROW_MUX_EN_0to7_SLOW);
	send_to_latch(4,ROW_MUX_EN_8to15_SLOW);
    // just delay, there is no need to MPU pulse.
    delayMicroseconds(SLOW_PULSE_WIDTH - 20);
	
	//restore rows to previous MUX condition,TODO: global variable, not really previous condition?
	send_to_latch(1,0);//Row0to7 A0 Address Lines
	send_to_latch(5,0);//Row8to15 A0 Address Lines
	send_to_latch(0,ROW_MUX_EN_0to7);//Enables 1-8 On
	send_to_latch(4,ROW_MUX_EN_8to15);//Enables 8-15 On
	//restore columns
    select_column_board(CB_NUMBER);
	send_to_latch(0,COL_MUX_EN);
  }

}

int dpe_read() {//fast read first several column boards, batch version of fast_read()
  if (Serial_println_ON == 1) {
    Serial.println("Enter total number of column board (1 to 8) to read from (NUM_COL_BOARDS)");
  }
  //Serial.println("go");
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  if (NUM_COL_BOARDS == 1024) {
    return -1;
  }

  // setup ADCs on all column boards.
  for (int MY_COUNTER1 = 0; MY_COUNTER1 < NUM_COL_BOARDS; MY_COUNTER1++) {
    select_column_board(MY_COUNTER1);
	send_to_latch(6,7);//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
  }

  // send the start pulse for fast reading.
  digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
  delayMicroseconds(5);
  digitalWrite(MPU_PULSE, LOW);

  for (int MY_COUNTER1 = 0; MY_COUNTER1 < NUM_COL_BOARDS; MY_COUNTER1++) {
    select_column_board(MY_COUNTER1);
    //Preselect latch with ADC control lines
	send_to_latch(6,7);//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH

    digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
    delayMicroseconds(5);
    digitalWrite(MPU_PULSE, LOW);

    //Probably should look at "END OF CONVERT" signal now, but it happens so fast
    //that I'm just putting in a small delay before doing reads.
    select_column_board(CB_NUMBER);
    //delayMicroseconds(1);
    PORTE = 5;//Lower ADC Not Chip Select
    //Do 8 Reads to Return ADC Values
    digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
      PORTE = 5;//Raise ADC_NOT_READ
      PORTE = 1;//Lower ADC_NOT_READ
      delayMicroseconds(1);
      ADC_READ_VALUE[MY_COUNTER0] = PORTD;
    }
    PORTE = 5;//Raise ADC_NOT_READ
    PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
    digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches

    for (int MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
      Serial.print(ADC_READ_VALUE[MY_COUNTER0]);
      Serial.print(",");
    }
    Serial.println(ADC_READ_VALUE[7]);
  }
  //if(Serial_println_ON == 1) {
  Serial.println("DPE read complete");
  //}
  return 0;

}

void batch_set_slow() {//set several lowercorner 16x8 subarray
  // Read in batch reset configurations - begin.
  if (Serial_println_ON == 1) {
    Serial.println("Batch reset slow command 1: number of column boards(1~8), number of row boards(1~8), slow column pulse width, COL_DAC_BATCH_SPAN(5 or 10), bottom_flag");
  }
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  SLOW_PULSE_WIDTH = constrain(SLOW_PULSE_WIDTH, 10, 1000000);
  while (Serial.available() == 0) {}
  COL_DAC_BATCH_SPAN = Serial.parseInt();
  // while (Serial.available() == 0) {}
  // bottom_flag = Serial.parseInt();

  int TOTAL_COLS = 8 * NUM_COL_BOARDS;
  int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;

  // if (TOTAL_ROWS > 64) {
  //   TOTAL_ROWS = 64;
  // }
  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",");
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("SLOW_PULSE_WIDTH: ");
    Serial.print(SLOW_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    // Serial.print(",");
    // Serial.print("bottom_flag: ");
    // Serial.print(bottom_flag);
    Serial.println("");
  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command slow 2: V_BATCH_RESET for column boards. Array size is previously defined by number of column and row boards.");
  }
  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      // To save memory, use V_BATCH_SET array for both set and reset operations.
      V_BATCH_SET[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command 3: V_BATCH_RESET_GATE for column boards. Array size is previously defined by number of column and row boards.");
  }

  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      // To save memory, use V_BATCH_SET_GATE array for both set and reset operations.
      V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  // Read in Batch reset configurations - complete.

  // Step 1:
  // loop setup all existing column boards:
  // Column MUX configuration table:latch0-EN,1-A0,2-A1
  // States:     Float Float Float Float Ground Slow_pulse Read Fast_pulse
  // COL_MUX_A0:   0     1     0     1      0       1        0      1
  // COL_MUX_A1:   0     0     1     1      0       0        1      1
  // COL_MUX_EN:   0     0     0     0      1       1        1      1
  if (Serial_println_ON == 1) {
    Serial.println("Start loop setup all existing column boards");
  }
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2 = MY_COUNTER2 + 1) {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in fast pulse mode: 0, 255, 255
	//floating all cols
	send_to_latch(0,0);//EN
	send_to_latch(1,0);//A0
	send_to_latch(2,0);//A1
	//set column gate and pulse width before double loop
    //No need to setup TIAs since it is in reset mode********************************
    setup_col_pulse_width(COL_BATCH_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);//No need to read so no need to set const etc.
    //setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    //setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    //setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);
  }

  if (Serial_println_ON == 1) {
    Serial.println("Complete Loop setup all existing column boards");
  }
  // Step 2:
  // loop setup all existing row boards:
  // floating all rows
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:latch0-EN,1-A0,2-A1,for row 8-15 latch+4
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in grounding mode.
	
	send_to_latch(0,0);//EN 0to7
	send_to_latch(1,0);//A0 0to7
	send_to_latch(2,0);//A1 0to7
	send_to_latch(4,0);//EN 8to15
	send_to_latch(5,0);//A0 8to15
	send_to_latch(6,0);//A1 8to15

    //setup_row_pulse_width(1024, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    //setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  if (Serial_println_ON == 1) {
    Serial.println("Loop setup all existing row boards");
  }
  // setup all column boards and row boards before batch reset - end.

  // Begin batch reset:
		//TODO row outer or col outer, which is better?
  for (int MY_COUNTER2 = 0; MY_COUNTER2 <  TOTAL_ROWS; MY_COUNTER2++) { // column-wise loop
    for (int MY_COUNTER1 = 0; MY_COUNTER1 <  TOTAL_COLS; MY_COUNTER1++) { // row-wise loop
      int current_row = MY_COUNTER2;
      int current_row_in_rb = current_row % 16;
	  int current_rb_number = current_row / 16;
      // if (bottom_flag > 0) {
		  // current_rb_number += 4;
      // } 

      int current_col = MY_COUNTER1;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

			if (V_BATCH_SET[current_row][current_col] != 0)
			{
		  //enable the row
        select_row_board(current_rb_number);
        if (current_row_in_rb < 8) {//TODO is there any side effects enabling other rows and cols when set one point here?
		  send_to_latch(0,PORTE_conversion_for_single_channel(current_row_in_rb));
        } else {
		  send_to_latch(4,PORTE_conversion_for_single_channel(current_row_in_rb - 8));
        }

        // setup the gate voltage for the selected column.
        select_column_board(current_cb_number);

        //Enable the selected column
        //Set-up Column MUX_A0 and MUX_A1********************************************
		send_to_latch(1,PORTE_conversion_for_single_channel(current_col_in_cb));
		send_to_latch(0,PORTE_conversion_for_single_channel(current_col_in_cb));
        //set column voltage and gate. Use DAC input as the slow pulse trigger.
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);

        //digitalWrite(SLOW_PULSE, HIGH);//Send Pulse That Starts the Entire Process
        delayMicroseconds(SLOW_PULSE_WIDTH);
        //digitalWrite(SLOW_PULSE, LOW);
		//reset column voltage and gate to 0
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
		
		send_to_latch(1,0);//Col A0 Address Lines
		send_to_latch(0,0);//MUX EN Address Lines
		
        select_row_board(current_rb_number);
		send_to_latch(0,0);//ROW_MUX_EN Address Lines
		send_to_latch(4,0);//ROW_MUX_EN Address Lines

        //disbale selected row by put ROW_MUX_EN to 0********************************************
      }
    }
  }

  safeStartAllRowsAndColumns();

  Serial.println("cmd17 Batch set slow complete");

}

void batch_read() {
	//TODO this function is a good example to learn from
	// Read in batch read configurations - begin.
	if (Serial_println_ON == 1)
	{
		Serial.println("Batch read command: V_read(0~5), V_gate(0~5), TIA gain number(1~4), pulse width(0~1023), S/H delay(0~1023), ADC convst delay(0~1023), number of column boards(1~8), number of row boards(1~8), bottom_flag");
	}
	while (Serial.available() == 0)
	{
	}
	V_BATCH_READ = Serial.parseFloat();
	while (Serial.available() == 0)
	{
	}
	V_BATCH_GATE = Serial.parseFloat();
	while (Serial.available() == 0)
	{
	}
	TIA_BATCH_GAIN = Serial.parseInt();
	while (Serial.available() == 0)
	{
	}
	ROW_BATCH_PULSE_WIDTH = Serial.parseInt();
	while (Serial.available() == 0)
	{
	}
	COL_BATCH_SH_DELAY = Serial.parseInt();
	while (Serial.available() == 0)
	{
	}
	COL_BATCH_AD_CONVST_DELAY = Serial.parseInt();
	while (Serial.available() == 0)
	{
	}
	NUM_COL_BOARDS = Serial.parseInt();
	while (Serial.available() == 0)
	{
	}
	NUM_ROW_BOARDS = Serial.parseInt();
	//while (Serial.available() == 0) {}
	//int bottom_flag = Serial.parseInt();
	// Read in batch read configurations - complete.
	int TOTAL_COLS = 8 * NUM_COL_BOARDS;
	int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;
	int ADC_BATCH_READ_VALUE[TOTAL_ROWS][TOTAL_COLS]; //Maximum array size.

	// if (TOTAL_ROWS > 64) {
	// 	TOTAL_ROWS = 64;
	// }

  if (Serial_println_ON == 1) {
    Serial.print(V_BATCH_READ);
    Serial.print(", ");
    Serial.print(V_BATCH_GATE);
    Serial.print(", ");
    Serial.print(TIA_BATCH_GAIN);
    Serial.print(", ");
    Serial.print(ROW_BATCH_PULSE_WIDTH);
    Serial.print(", ");
    Serial.print(COL_BATCH_SH_DELAY);
    Serial.print(", ");
    Serial.print(COL_BATCH_AD_CONVST_DELAY);
    Serial.print(", ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(", ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(", ");
    Serial.print(TOTAL_COLS);
    Serial.print(", ");
    Serial.print(TOTAL_ROWS);
    Serial.println("");
  }


  // Setup column boards and row boards before loop reading.

  switch (TIA_BATCH_GAIN) {
    case 1:
      TIA_BATCH_READ_A0 = 0;
      TIA_BATCH_READ_A1 = 0;
      break;
    case 2:
      TIA_BATCH_READ_A0 = 255;
      TIA_BATCH_READ_A1 = 0;
      break;
    case 3:
      TIA_BATCH_READ_A0 = 0;
      TIA_BATCH_READ_A1 = 255;
      break;
    case 4:
      TIA_BATCH_READ_A0 = 255;
      TIA_BATCH_READ_A1 = 255;
      break;
    default:
      TIA_BATCH_READ_A0 = 0;
      TIA_BATCH_READ_A1 = 0;
      break;
  }

  // loop setup all existing column boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2++) {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in reading mode: 0, 0, 255
	//floating all columns
	send_to_latch(0,0);//MUX Enable Lines
	send_to_latch(1,0);//MUX A0 Address Lines
	send_to_latch(2,255);//MUX A1 Address Lines
	send_to_latch(4,TIA_BATCH_READ_A0);//TIA_A0 Set-up TIA gain
	send_to_latch(5,TIA_BATCH_READ_A1);//TIA_A1
	//set column voltage(0) and gate(0) and adc
    setup_col_pulse_width(COL_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);
  }

  // loop setup all existing row boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in fast_pulse mode.
    //set up ROW_MUX_EN_0to7
	send_to_latch(0,0);//EN
	send_to_latch(1,255);//A0
	send_to_latch(2,255);//A1

    //set up ROW_MUX_EN_8to15
	send_to_latch(4,0);//EN
	send_to_latch(5,255);//A0
	send_to_latch(6,255);//A1
    for (int MY_COUNTER0 = 0; MY_COUNTER0 <  8; MY_COUNTER0++) {
      ROW_DAC_BATCH_VOLTAGE_0to7[MY_COUNTER0] = V_BATCH_READ;
      ROW_DAC_BATCH_VOLTAGE_8to15[MY_COUNTER0] = V_BATCH_READ;
    }
	//set row voltage and time
    setup_row_pulse_width(ROW_BATCH_PULSE_WIDTH, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
	//TODO row 0~7 amd 8~15 can be combined if they always appear together
    setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  // setup all column boards and row boards before batch read - end.

  // Begin batch read:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 <  TOTAL_COLS; MY_COUNTER2++) { // column-wise loop
    int current_col = MY_COUNTER2;
    int current_col_in_cb = current_col % 8;
    int current_cb_number = current_col / 8;

    select_column_board(current_cb_number);

    // Fast read for the selected line: COL_MUX = selected line high, COL_MUX_A0 = 255 (unchanged), COL_MUX_A1 = 255 (unchanged);
	send_to_latch(0,PORTE_conversion_for_single_channel(current_col_in_cb));//EN

    // set column gate Turn on the selected transistor line.
    //setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
    setup_col_sel_voltage_oneline(V_BATCH_GATE, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
    //COL_SEL_BATCH_VOLTAGE_ZERO[current_col_in_cb] = V_BATCH_GATE;
    //setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, current_cb_number, COL_DAC_BATCH_SPAN);
    //COL_SEL_BATCH_VOLTAGE_ZERO[current_col_in_cb] = 0;

    for (int MY_COUNTER1 = 0; MY_COUNTER1 <  TOTAL_ROWS; MY_COUNTER1++) { // row-wise loop
      int current_row = MY_COUNTER1;
      int current_row_in_rb = current_row % 16;

	  int current_rb_number = current_row / 16;
      // Check if the current_row_in_rb < 8
      select_row_board(current_rb_number);
      if (current_row_in_rb < 8) {
		send_to_latch(0,PORTE_conversion_for_single_channel(current_row_in_rb));//EN
      } else {
		send_to_latch(4,PORTE_conversion_for_single_channel(current_row_in_rb - 8));//EN
      }

      //fast read as in option 4
      select_column_board(current_cb_number);
      //Preselect latch with ADC control lines
	  send_to_latch(6,7);//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH

      delayMicroseconds(1);
      digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
      delayMicroseconds(5);
      digitalWrite(MPU_PULSE, LOW);

      select_column_board(current_cb_number);
      delayMicroseconds(1);
      PORTE = 5;//Lower ADC Chip Select
      //Do 8 Reads to Return ADC Values
      digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches

      //for(MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
      //PORTE = 1;//Lower ADC_NOT_READ
      //ADC_READ_VALUE[MY_COUNTER0] = PORTD;
      //PORTE = 5;//Raise ADC_NOT_READ
      ////delayMicroseconds(1);
      //}

      for (int MY_COUNTER0 = 0; MY_COUNTER0 < 8; MY_COUNTER0++) {
        PORTE = 5;//Raise ADC_NOT_READ
        //delayMicroseconds(1);
        PORTE = 1;//Lower ADC_NOT_READ
        delayMicroseconds(1);
        ADC_READ_VALUE[MY_COUNTER0] = PORTD;
        //PORTE = 5;//Raise ADC_NOT_READ
        //delayMicroseconds(1);
      }
      PORTE = 5;//Raise ADC_NOT_READ

      PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels

      digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches

      // clean the shifting problem.
      //TEMP1 = ADC_READ_VALUE[0];
      //for(MY_COUNTER0 = 0; MY_COUNTER0 < 7; MY_COUNTER0++) {
      //    ADC_READ_VALUE[MY_COUNTER0] = ADC_READ_VALUE[MY_COUNTER0+1];
      //}
      //ADC_READ_VALUE[7] = TEMP1;

      // Save the read data in ADC_BATCH_READ_VALUE, also clean the shifting problem.
	  ADC_BATCH_READ_VALUE[MY_COUNTER1][MY_COUNTER2] = ADC_READ_VALUE[current_col_in_cb];
      switch (current_row_in_rb) {
        // Change the selected row back to ground if current_row_in_rb == 7 or 15.
        case 7:
          select_row_board(current_rb_number);
		  send_to_latch(0,0);
          break;
        case 15:
          select_row_board(current_rb_number);
		  send_to_latch(4,0);
          break;
        default:
          break;
      }
    }
	//disable all cloumns in the end
    // change the selected column back to ground if current_col_in_cb == 7.
    if (current_col_in_cb == 7) {
      select_column_board(current_cb_number);
	  send_to_latch(0,0);
    }
    // Turn off the selected transistor line.
    //COL_DAC_BATCH_SPAN = 5;
    setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);//TODO is this not necessary?
    setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
    //setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, current_cb_number, COL_DAC_BATCH_SPAN);
  }

  safeStartAllRowsAndColumns();

  // Print out raw ADC data col by col.
  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1++) {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0++) {
      Serial.print(ADC_BATCH_READ_VALUE[MY_COUNTER1][MY_COUNTER0]);
      Serial.print(",");
    }
    Serial.println(ADC_BATCH_READ_VALUE[MY_COUNTER1][TOTAL_COLS - 1]);
  }
  Serial.println("Batch read complete");

}

void Batch_Reset_Slow() {
  // Read in batch set configurations - begin.
  if (Serial_println_ON == 1)
  {
    Serial.println("Batch set command 1: number of column boards(1~8), number of row boards(1~8), slow row pulse width, bottom_flag");
  }
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  // while (Serial.available() == 0) {}
  // int bottom_flag = Serial.parseInt();

  COL_DAC_BATCH_SPAN = 10;
  ROW_DAC_BATCH_SPAN = 10;

  int TOTAL_COLS = 8 * NUM_COL_BOARDS;
  int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;
  // if (TOTAL_ROWS > 64)
  // {
  //   TOTAL_ROWS = 64;
  // }
  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1)
  {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",");
    Serial.print("ROW_SLOW_PULSE_WIDTH: ");
    Serial.print(SLOW_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    Serial.print(",");
    Serial.print("ROW_DAC_BATCH_SPAN: ");
    Serial.print(ROW_DAC_BATCH_SPAN);
    // Serial.print(",");
    // Serial.print("bottom_flag: ");
    // Serial.println(bottom_flag);
  }

  if (Serial_println_ON == 1) Serial.println("Batch set command 2: V_BATCH_SET for row boards. Array size is previously defined by number of column and row boards.");
  // Read voltage signal row by row;
  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1)
  {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1)
    {
      while (Serial.available() == 0) {}
      V_BATCH_SET[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1)
  {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1)
    {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1)
      {
        Serial.print(V_BATCH_SET[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }
  if (Serial_println_ON == 1) Serial.println("Batch set command 3: V_BATCH_SET_GATE for column boards. Array size is previously defined by number of column and row boards.");


  for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1)
  {
    for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1)
    {
      while (Serial.available() == 0) {}
      V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1)
  {
    for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1)
    {
      for (int MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1)
      {
        Serial.print(V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  // Step 1:
  // loop setup all existing column boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2++)
  {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in reading mode: 0, 0, 255
    // Here the MUX are all disabled, and the all columns are grounded.
	send_to_latch(0,0);
	send_to_latch(1,0);
	send_to_latch(2,0);

    //No need to setup TIAs since it is in set mode********************************
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN); // set all column DAC selector votlage to 0.

  }

  if (Serial_println_ON == 1)
  {
    Serial.println("Loop setup all existing column boards complete.");
  }

  // Step 2:
  // loop setup all existing row boards:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++)
  {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in fast_pulse mode.
    //set up ROW_MUX_EN_0to7
	send_to_latch(0,0);
	send_to_latch(1,0);
	send_to_latch(2,0);
    //set up ROW_MUX_EN_8to15
	send_to_latch(4,0);
	send_to_latch(5,0);
	send_to_latch(6,0);

    setup_row_pulse_width(1000, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    //setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  if (Serial_println_ON == 1)
  {
    Serial.println("Loop setup all existing row boards complete.");
  }
  // setup all column boards and row boards before batch set - end.

  // Begin batch set:
  for (int MY_COUNTER2 = 0; MY_COUNTER2 <  TOTAL_COLS; MY_COUNTER2++)
  { // column-wise loop
    for (int MY_COUNTER1 = 0; MY_COUNTER1 <  TOTAL_ROWS; MY_COUNTER1++)
    { // row-wise loop

      int current_col = MY_COUNTER2;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

      int current_row = MY_COUNTER1;
      int current_row_in_rb = current_row % 16;
	  int current_rb_number = current_row / 16;
      // if (bottom_flag > 0) {
      //   current_rb_number +=4;
      // }

      if (V_BATCH_SET[current_row][current_col] > 0)
      {

        select_column_board(current_cb_number);
		send_to_latch(0,PORTE_conversion_for_single_channel(current_col_in_cb));//EN
        //select_column_board(current_cb_number);
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
		//TODO: shouln't we select row board first here??!! contained selection in those setup function
          setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);//TODO should be able to move out?
          setup_row_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
        select_row_board(current_rb_number);
        if (current_row_in_rb < 8)
        {
//          // Change to the selected row in the first 8 rows:
//          setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
//          setup_row_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);

          select_row_board(current_rb_number);
		  send_to_latch(1,PORTE_conversion_for_single_channel(current_row_in_rb));//A0
		  send_to_latch(0,PORTE_conversion_for_single_channel(current_row_in_rb));//EN
        }
        else
        {
          //select_row_board(current_rb_number);
//          // Change to the selected row in the last 8 rows:
//          setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
//          setup_row_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);

          select_row_board(current_rb_number);
		  
		  send_to_latch(5,PORTE_conversion_for_single_channel(current_row_in_rb - 8));//A0
		  send_to_latch(4,PORTE_conversion_for_single_channel(current_row_in_rb - 8));//EN
        }

        delayMicroseconds(SLOW_PULSE_WIDTH);

        //select_row_board(current_rb_number);
        if (current_row_in_rb < 8)
        {
          // Change to the selected row in the first 8 rows:
		  send_to_latch(1,0);//A0
		  send_to_latch(0,0);//EN
          select_column_board(current_cb_number);
		  send_to_latch(0,0);//EN
          setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
          setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);

          select_row_board(current_rb_number);
          setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
          setup_row_dac_voltage_oneline(0, current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
        }
        else
        {
			
		  send_to_latch(5,0);//A0
		  send_to_latch(4,0);//EN

          select_column_board(current_cb_number);
		  send_to_latch(0,0);//EN
          setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
          setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);

          select_row_board(current_rb_number);
          setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
          setup_row_dac_voltage_oneline(0, current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
        }
      }
    }
  }

  safeStartAllRowsAndColumns();
  // Print task complete message.
  Serial.println("cmd18 Reset batch slow complete");
}

void Arbitrary_Wave() {
  int DATA_POINTS;
  int TOTAL_ROWS;
  int ENABLES[16];

  while (Serial.available() == 0) {}
  ROW_PULSE_WIDTH = Serial.parseInt();

  while (Serial.available() == 0) {}
  ROW_DAC_SPAN = Serial.parseInt();

  while (Serial.available() == 0) {}
  TOTAL_ROWS = Serial.parseInt();
  TOTAL_ROWS = constrain(TOTAL_ROWS, 1, 128);
  int ACTIVE_ROW_NUMBER[TOTAL_ROWS];

  while (Serial.available() == 0) {}
  DATA_POINTS = Serial.parseInt();
  float DAC_VOLTAGES[TOTAL_ROWS][DATA_POINTS];

  while (Serial.available() == 0) {}
  for (int i = 0; i < 16; i++){
    ENABLES[i] = Serial.parseInt();
  }

  while (Serial.available() == 0) {}
  for (int row = 0; row < TOTAL_ROWS; row++){
    ACTIVE_ROW_NUMBER[row] = Serial.parseInt();
  }

  while (Serial.available() == 0) {}
  for (int row = 0; row < TOTAL_ROWS; row++){
    for (int point = 0; point < DATA_POINTS; point++){
      DAC_VOLTAGES[row][point] = Serial.parseFloat();
    }
  }

  for (int boardnumber = 0; boardnumber < 8; boardnumber++){
    int BOARD_NUMBER = boardnumber;
    setup_row_dac_span(ROW_DAC_SPAN, BOARD_NUMBER);
    //    setup_row_dac_toggle_select(BOARD_NUMBER, true, true);
    setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, BOARD_NUMBER, ROW_DAC_SPAN);
    setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, BOARD_NUMBER, ROW_DAC_SPAN);
    //    setup_row_dac_toggle_global(BOARD_NUMBER, true, false);
    //    setup_row_dac_toggle_pin(boardnumber, true);

    select_row_board(BOARD_NUMBER);
	send_to_latch(0,ENABLES[BOARD_NUMBER * 2]);//EN
	send_to_latch(1,255);//A0
	send_to_latch(2,0);//A1
	
	send_to_latch(4,ENABLES[BOARD_NUMBER * 2 + 1]);//EN
	send_to_latch(5,255);//A0
	send_to_latch(6,0);//A1
	
	send_to_latch(3,B00101011);//43

  }

  digitalWrite(TRIG, HIGH);
  for (int point = 0; point < DATA_POINTS; point++) {
    //Serial.println(micros());
    //    Serial.println();
    //    Serial.print("point =");
    //    Serial.println(point);
    for (int row = 0; row < TOTAL_ROWS; row++){
      int row_number = ACTIVE_ROW_NUMBER[row];
      int row_board_number = row_number / 16;
      int row_number_in_rb = row_number % 16;
      float voltage = DAC_VOLTAGES[row][point];
      setup_row_dac_voltage_oneline2(voltage, row_number_in_rb, row_board_number, ROW_DAC_SPAN);
    }
    for (int boardnumber = 0; boardnumber < 8; boardnumber++){
      setup_row_dac_update_all(boardnumber);
    }
  }
  digitalWrite(TRIG, LOW);
  for (int boardnumber = 0; boardnumber < 8; boardnumber++){
    setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, boardnumber, ROW_DAC_SPAN);
    setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, boardnumber, ROW_DAC_SPAN);
  }
}

void setup_row_dac_update_all(int RB_NUMBER_TEMP) {
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  select_the_DAC();
  send_to_DAC(8, B10010000);
  send_to_DAC(16, B0); //
  deselect_the_DAC();
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}

void setup_row_dac_voltage_oneline2(float ROW_DAC_VOLTAGE_0to7_TEMP, int LINE_ADDRESS, int RB_NUMBER_TEMP, int ROW_DAC_SPAN_TEMP) {//B0000
  int DAC_address_bit;
  int DAC_voltage_bit;
  select_row_board(RB_NUMBER_TEMP);
  select_latch(7);
  digitalWrite(NOT_BOARD_OE[0], LOW);//Allow writing to latches
  DAC_address_bit = LINE_ADDRESS;
  DAC_voltage_bit = int ((ROW_DAC_VOLTAGE_0to7_TEMP + (float)ROW_DAC_SPAN_TEMP) / ((float)ROW_DAC_SPAN_TEMP * 2) * (float)65536);
  select_the_DAC();
  //send command - 4 bits
  send_to_DAC(4, B0000);
  //send address - 4 bits
  send_to_DAC(4, DAC_address_bit);
  //send value - 16 bits (sending 32768/2 which should be 2.5V on 0-5V range
  send_to_DAC(16, DAC_voltage_bit);
  deselect_the_DAC();
  digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches.
}
// Global parameters (for Case 20, 40, 41)
int DATA_ROWS; // No of samples of the batch
int ROW_ENABLES[16]; // Row enable vector (1: enable; 0: disable)
int COL_ENABLES[8]; // Column enable vector (1: enable; 0: disable)
float GATE_VOLTAGE[64]; // Gate voltage (of col 1 to 64)
// int REPEAT; // No of repeats of the batch
//int ONLY_GET_VOL; // DPE with previous inputs
float GATE_VOLTAGE_TEMP[8];
float voltage;
float voltage_divider = 630;  // The transfered data is int, divided by this number to get voltages.

int Batch_DPE() // The main sub function
{
	// while (Serial.available() == 0) {} // Wait for nonzero input buffer
	// ONLY_GET_VOL = Serial.parseInt(); // 

	// if (ONLY_GET_VOL == 0)
	// {
		// while (Serial.available() == 0) {} // Wait for nonzero input buffer
		// int REPEAT = Serial.parseInt(); // Times to repeat

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		ROW_PULSE_WIDTH = Serial.parseInt(); // Row pulse width

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		ROW_DAC_SPAN = Serial.parseInt(); // DAC range

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		TOTAL_ROWS = Serial.parseInt(); // Total rows

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		TOTAL_COLS = Serial.parseInt(); // Total coloumns

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		TIA_BATCH_GAIN = Serial.parseInt(); // TIA Gain select

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		DATA_ROWS = Serial.parseInt(); //all samples in the batch

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		for (int i = 0; i < 16; i++)
		{
			ROW_ENABLES[i] = Serial.parseInt(); //Which row to enable
		}

		while (Serial.available() == 0) {} // Wait for nonzero input buffer
		for (int i = 0; i < 8; i++)
		{
			COL_ENABLES[i] = Serial.parseInt(); //Which column to enable
		}

		for (int i = 0; i < TOTAL_COLS; i++)
		{
			while (Serial.available() == 0) {}  // Wait for nonzero input buffer
			GATE_VOLTAGE[i] = Serial.parseFloat(); //Gate voltages over columns
		}
	// }

	char APPLIED_VOLTAGES[DATA_ROWS*TOTAL_ROWS];
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES, 1); //dump the separator
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES, DATA_ROWS * TOTAL_ROWS); //read the entire input

	if (Serial_println_ON == 1) // In case to print out the parameters
	{
		// Serial.print("ONLY_GET_VOL: ");
		// Serial.println(ONLY_GET_VOL);

		// Serial.print("REPEAT: ");
		// Serial.println(REPEAT);

		Serial.print("ROW_PULSE_WIDTH: ");
		Serial.println(ROW_PULSE_WIDTH);

		Serial.print("ROW_DAC_SPAN: ");
		Serial.println(ROW_DAC_SPAN);

		Serial.print("TOTAL_ROWS: ");
		Serial.println(TOTAL_ROWS);

		Serial.print("TOTAL_COLS: ");
		Serial.println(TOTAL_COLS);

		Serial.print("TIA_BATCH_GAIN: ");
		Serial.println(TIA_BATCH_GAIN);

		Serial.print("DATA_ROWS: ");
		Serial.println(DATA_ROWS);

		Serial.print("ROW_ENABLES: ");
		for (int i = 0; i < 15; i++)
		{
			Serial.print(ROW_ENABLES[i]);
			Serial.print(',');
		}
		Serial.println(ROW_ENABLES[15]);

		Serial.print("COL_ENABLES: ");
		for (int i = 0; i < 7; i++)
		{
			Serial.print(COL_ENABLES[i]);
			Serial.print(',');
		}
		Serial.println(COL_ENABLES[7]);

		Serial.print("GATE_VOLTAGE: ");
		for (int i = 0; i < TOTAL_COLS; i++)
		{
			Serial.print(GATE_VOLTAGE[i]);
			Serial.print(',');
		}
		Serial.println();

		Serial.println("APPLIED_VOLTAGES: ");
		for (int datarows = 0; datarows < DATA_ROWS; datarows++) // repeater for all samples in the batch
		{
			for (int row = 0; row < TOTAL_ROWS; row++)
			{
				Serial.print((float)APPLIED_VOLTAGES[datarows*TOTAL_ROWS + row] / voltage_divider, 4);
				Serial.print(',');
			}
			Serial.println();
		}
		Serial.println("Upload end! ");
		return -1;
	}

	// if (ONLY_GET_VOL == 0)
	// {
		// Gain settings
		TIA_Gain_Convert(TIA_BATCH_GAIN);

		// Row boards initializatoin
		for (int ROW_BOARD_NUMBER = 0; ROW_BOARD_NUMBER < 8; ROW_BOARD_NUMBER++)
		{

			select_row_board(ROW_BOARD_NUMBER); //
			select_latch(0);
			PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2]; //
			pulse_the_latch();
			select_latch(1); // Row A0 Address Lines
			PORTE = 255;
			pulse_the_latch();
			select_latch(2); // Row A1 Address Lines
			PORTE = 255;
			pulse_the_latch();

			select_latch(4);
			PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2 + 1];
			pulse_the_latch();
			select_latch(5); //Row A0 Address Lines
			PORTE = 255;
			pulse_the_latch();
			select_latch(6); //Row A1 Address Lines
			PORTE = 255;
			pulse_the_latch();

			setup_row_pulse_width(ROW_PULSE_WIDTH, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_row_dac_span(ROW_DAC_SPAN, ROW_BOARD_NUMBER);
			setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
		}

		// Column boards initialization
		for (int COL_BOARD_NUMBER = 0; COL_BOARD_NUMBER < 8; COL_BOARD_NUMBER++)
		{
			select_column_board(COL_BOARD_NUMBER);
			select_latch(0); //MUX Enable Lines
			PORTE = COL_ENABLES[COL_BOARD_NUMBER]; //
			pulse_the_latch();

			select_latch(1); //MUX A0 Address Lines
			PORTE = 0;
			pulse_the_latch();

			select_latch(2); //MUX A1 Address Lines
			PORTE = 255;
			pulse_the_latch();

			select_latch(4); //TIA A0 Address Lines
			PORTE = TIA_BATCH_READ_A0;
			pulse_the_latch();

			select_latch(5); //TIA A1 Address Lines
			PORTE = TIA_BATCH_READ_A1;
			pulse_the_latch();

			for (int i = 0; i < 8; i++)
			{
				GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE[i + COL_BOARD_NUMBER * 8];
			}

			setup_col_pulse_width(ROW_PULSE_WIDTH + 20, COL_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_col_sample_and_hold(ROW_PULSE_WIDTH - 100, COL_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_col_adc_convst(ROW_PULSE_WIDTH + 10, COL_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_col_dac_span(ROW_DAC_SPAN, COL_BOARD_NUMBER);
			setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, COL_BOARD_NUMBER, ROW_DAC_SPAN);
			setup_col_sel_span(COL_SEL_SPAN, COL_BOARD_NUMBER);
			setup_col_sel_voltage(GATE_VOLTAGE_TEMP, COL_BOARD_NUMBER, COL_SEL_SPAN);
		}
	// }

	// Start DPE
	// for (int repeat = 0; repeat < REPEAT; repeat++)
	// {

		// Visit all samples of the batch
		for (int data_row = 0; data_row < DATA_ROWS; data_row++)
		{
			
			// Rows of each sample (a column vector)
			for (int row = 0; row < TOTAL_ROWS; row++)
			{
				int current_rb_number = row / 16;
				int current_row_in_rb = row % 16;
				voltage = (float)APPLIED_VOLTAGES[data_row *TOTAL_ROWS + row] / voltage_divider;
				setup_row_dac_voltage_oneline(voltage, current_row_in_rb, current_rb_number, ROW_DAC_SPAN);
			}

			// Send voltages
			digitalWrite(MPU_PULSE, HIGH); // Send Pulse That Starts the Entire Process
			delayMicroseconds(5); // Manual timer
			digitalWrite(MPU_PULSE, LOW); //

			// Receive data on columns
			for (int CB_NUMBER = 0; CB_NUMBER < TOTAL_COLS / 8; CB_NUMBER++)
			{
				select_column_board(CB_NUMBER);
				select_latch(6); //Preselect latch with ADC control lines
				PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
				pulse_the_latch();
				PORTE = 5;//Lower ADC Not Chip Select
				digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
				for (int i = 0; i < 8; i++)
				{
					PORTE = 5;//Raise ADC_NOT_READ
					PORTE = 1;//Lower ADC_NOT_READ
					delayMicroseconds(1);
					ADC_READ_VALUE[i] = PORTD;

					int16_t temp = (int16_t)ADC_READ_VALUE[i];
					Serial.write((char *)&temp, 2); // Advantage (no memory occupation)
				}
				PORTE = 5;//Raise ADC_NOT_READ
				PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels     
				digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches

			}
		}
		Serial.println("end");
	// }
	return 1;
}

void TIA_Gain_Convert(int TIA_BATCH_GAIN)
{
	switch (TIA_BATCH_GAIN)
	{
	case 1:
		TIA_BATCH_READ_A0 = 0;
		TIA_BATCH_READ_A1 = 0;
		break;
	case 2:
		TIA_BATCH_READ_A0 = 255;
		TIA_BATCH_READ_A1 = 0;
		break;
	case 3:
		TIA_BATCH_READ_A0 = 0;
		TIA_BATCH_READ_A1 = 255;
		break;
	case 4:
		TIA_BATCH_READ_A0 = 255;
		TIA_BATCH_READ_A1 = 255;
		break;
	default:
		TIA_BATCH_READ_A0 = 0;
		TIA_BATCH_READ_A1 = 0;
		break;
	}
}

// Case 31
int batch_reset_monovoltage() {
	if (Serial_println_ON == 1) {
		Serial.println("Batch reset command 1: number of column boards(1~8), number of row boards(1~8), row pulse width(0~1023), COL_DAC_BATCH_SPAN, ROW_DAC_BATCH_SPAN, Reset voltage(1-3V), Gate voltage(0-8V)");
	}
	while (Serial.available() == 0) {}
	NUM_COL_BOARDS = Serial.parseInt();
	while (Serial.available() == 0) {}
	NUM_ROW_BOARDS = Serial.parseInt();
	while (Serial.available() == 0) {}
	ROW_BATCH_PULSE_WIDTH = Serial.parseInt();
	while (Serial.available() == 0) {}
	COL_DAC_BATCH_SPAN = Serial.parseInt();
	while (Serial.available() == 0) {}
	ROW_DAC_BATCH_SPAN = Serial.parseInt();
	while (Serial.available() == 0) {}
	float V_reset = Serial.parseFloat();
	while (Serial.available() == 0) {}
	float V_gate = Serial.parseFloat();

	TOTAL_COLS = 8 * NUM_COL_BOARDS;
	TOTAL_ROWS = 16 * NUM_ROW_BOARDS;

	// Verify if parameters are correctly received.
	if (Serial_println_ON == 1) {
		Serial.print("NUM_COL_BOARDS: ");
		Serial.println(NUM_COL_BOARDS);
		Serial.print(",");
		Serial.print("NUM_ROW_BOARDS: ");
		Serial.println(NUM_ROW_BOARDS);
		Serial.print(",");
		Serial.print("TOTAL_ROWS: ");
		Serial.println(TOTAL_ROWS);
		Serial.print(",");
		Serial.print("TOTAL_COLS: ");
		Serial.println(TOTAL_COLS);
		Serial.print(",");
		Serial.print("ROW_BATCH_PULSE_WIDTH: ");
		Serial.println(ROW_BATCH_PULSE_WIDTH);
		Serial.print(",");
		Serial.print("COL_DAC_BATCH_SPAN: ");
		Serial.println(COL_DAC_BATCH_SPAN);
		Serial.print(",");
		Serial.print("ROW_DAC_BATCH_SPAN: ");
		Serial.println(ROW_DAC_BATCH_SPAN);
		Serial.print(",");
		Serial.print("V_reset: ");
		Serial.println(V_reset);
		Serial.print(",");
		Serial.print("V_gate: ");
		Serial.println(V_gate);
	}

	if (Serial_println_ON == 1) {
		Serial.println("Batch reset command 2: V_BATCH_SET for row boards. Array size is previously defined by number of column and row boards.");
	}
	// Read voltage signal row by row;

	uint8_t mask;
	Serial.readBytes(&mask, 1);
	if (Serial_println_ON == 1) {
		Serial.println(mask);
		Serial.println("masks:");
	}

	for (int i = 0; i < TOTAL_ROWS; i = i + 1) {
		for (int j = 0; j < NUM_COL_BOARDS; j = j + 1) {
			Serial.readBytes(&mask, 1);

			if (Serial_println_ON == 1) {
				Serial.print(mask);
				Serial.print(',');
			}
			for (int b = 0; b < 8; b++) {
				if (mask & 0x80) {
					V_BATCH_SET[i][j * 8 + b] = V_reset;
				}
				else {
					V_BATCH_SET[i][j * 8 + b] = 0;
				}
				mask = mask << 1;
			}
		}

		if (Serial_println_ON == 1) {
			Serial.println();
		}
	}

	// Verify if parameters are correctly received.
	if (Serial_println_ON == 1) {
		for (int i = 0; i < TOTAL_ROWS; i = i + 1) {
			for (int j = 0; j < TOTAL_COLS - 1; j = j + 1) {
				Serial.print(V_BATCH_SET[i][j]);
				Serial.print(", ");
			}
			Serial.println(V_BATCH_SET[i][TOTAL_COLS - 1]);
		}
		Serial.println("All data print out.");

		return -1;
	}

	// Step 1:
	// loop setup all existing column boards:
	for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2++) {
		CB_BATCH_NUMBER = MY_COUNTER2;
		select_column_board(CB_BATCH_NUMBER);

		// Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
		// Floating all columns while they are all in reading mode: 0, 0, 255
		// Here the MUX are all disabled, and the all columns are grounded.
		//Set-up Column MUX Enable********************************************
		select_latch(0);//MUX Enable Lines
		PORTE = 0; //
		pulse_the_latch();

		//Set-up Column MUX_A0 and MUX_A1********************************************
		select_latch(1);//MUX A0 Address Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(2);//MUX A1 Address Lines
		PORTE = 0;
		pulse_the_latch();

		//No need to setup TIAs since it is in set mode********************************
		//setup_col_pulse_width(1024, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
		//setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
		//setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
		//setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
		//setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN); // set all column DAC device voltages to 0.
		//COL_SEL_BATCH_VOLTAGE_ZERO is always 0.
		setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
		setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN); // set all column DAC selector votlage to 0.

	}

	if (Serial_println_ON == 1) {
		Serial.println("Loop setup all existing column boards complete.");
	}

	// Step 2:
	// loop setup all existing row boards:
	for (int MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
		// Set-up all row boards - begin ***************************************************
		//Set up MUX on Row Board ********************************************************
		RB_BATCH_NUMBER = MY_COUNTER2;
		select_row_board(RB_BATCH_NUMBER);

		// Row MUX configuration table:
		// States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
		// ROW_MUX_A0:   0     1     0     1      0       1        0      1
		// ROW_MUX_A1:   0     0     1     1      0       0        1      1
		// ROW_MUX_EN:   0     0     0     0      1       1        1      1

		// Floating all rows while all in fast_pulse mode.
		//set up ROW_MUX_EN_0to7
		select_latch(0);//Row Enable Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(1);//Row A0 Address Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(2);//Row A1 Address Lines
		PORTE = 0;
		pulse_the_latch();

		//set up ROW_MUX_EN_8to15
		select_latch(4);//Row Enable Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(5);//Row A0 Address Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(6);//Row A1 Address Lines
		PORTE = 0;
		pulse_the_latch();

		setup_row_pulse_width(ROW_BATCH_PULSE_WIDTH, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
		//setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
		//setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
		//setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
	}
	if (Serial_println_ON == 1) {
		Serial.println("Loop setup all existing row boards complete.");
	}
	// setup all column boards and row boards before batch set - end.

	// Begin batch reset:
	for (int MY_COUNTER2 = 0; MY_COUNTER2 < TOTAL_ROWS; MY_COUNTER2++) { // column-wise loop
		for (int MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_COLS; MY_COUNTER1++) { // row-wise loop

      int current_col = MY_COUNTER1;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

      int current_row = MY_COUNTER2;

      int current_rb_number = current_row / 16;
      int current_row_in_rb = current_row % 16;
      //current_rb_number = current_row / 16;

			if (V_BATCH_SET[current_row][current_col] > 0) {

        select_row_board(current_rb_number);
        setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
        setup_row_dac_voltage_oneline(V_BATCH_SET[current_row][current_col], current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);

				select_column_board(current_cb_number);
				select_latch(0);//MUX EN Address Lines
				PORTE = PORTE_conversion_for_single_channel(current_col_in_cb);
				pulse_the_latch();

				setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
				setup_col_sel_voltage_oneline(V_gate, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
				delayMicroseconds(10);

				select_row_board(current_rb_number);
				if (current_row_in_rb < 8) {
					//select_row_board(current_rb_number);
					select_latch(1);//Row A0 Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb);
					pulse_the_latch();

					select_latch(2);//Row A1 Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb);
					pulse_the_latch();

					select_latch(0);//ROW_MUX_EN Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb);
					pulse_the_latch();
				}
				else {
					//select_row_board(current_rb_number);
					select_latch(5);//Row A0 Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb - 8);
					pulse_the_latch();

					select_latch(6);//Row A1 Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb - 8);
					pulse_the_latch();

					select_latch(4);//ROW_MUX_EN Address Lines
					PORTE = PORTE_conversion_for_single_channel(current_row_in_rb - 8);
					pulse_the_latch();
				}

				digitalWrite(MPU_PULSE, HIGH);
				delayMicroseconds(10);
				digitalWrite(MPU_PULSE, LOW);

				select_column_board(current_cb_number);
				select_latch(0);//MUX EN Address Lines
				PORTE = 0;
				pulse_the_latch();
				setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
				setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);

				select_row_board(current_rb_number);
				if (current_row_in_rb < 8) {
					// Change to the selected row in the first 8 rows:

					select_latch(1);//Row A0 Address Lines
					PORTE = 0;
					pulse_the_latch();

					select_latch(2);//Row A1 Address Lines
					PORTE = 0;
					pulse_the_latch();

					select_latch(0);//ROW_MUX_EN Address Lines
					PORTE = 0;
					pulse_the_latch();
				}
				else {
					select_latch(5);//Row A0 Address Lines
					PORTE = 0;
					pulse_the_latch();

					select_latch(6);//Row A1 Address Lines
					PORTE = 0;
					pulse_the_latch();

					select_latch(4);//ROW_MUX_EN Address Lines
					PORTE = 0;
					pulse_the_latch();
				}

        //select_row_board(current_rb_number);
        setup_row_dac_span(ROW_DAC_BATCH_SPAN, current_rb_number);
        setup_row_dac_voltage_oneline(0, current_row_in_rb, current_rb_number, ROW_DAC_BATCH_SPAN);
      }
    }
  }

  safeStartAllRowsAndColumns();
  // Print task complete message.
  Serial.println("cmd31 batch_reset_monovoltage complete");
  return 1;
}

int batch_set_row_voltage() //case 32
{
  float voltage_temp[8];

  float V_BATCH_SET_GATE[128][64];
  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  SLOW_PULSE_WIDTH = constrain(SLOW_PULSE_WIDTH, 10, 1000000);
  while (Serial.available() == 0) {}
  COL_DAC_BATCH_SPAN = Serial.parseInt();
  while (Serial.available() == 0) {}
  int set_flag = Serial.parseInt(); 
  while (Serial.available() == 0) {}
  float V_SET = Serial.parseFloat(); 

  TOTAL_COLS = 8 * NUM_COL_BOARDS;
  TOTAL_ROWS = 16 * NUM_ROW_BOARDS;

  if (Serial_println_ON == 1)
  {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",");
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("SLOW_PULSE_WIDTH: ");
    Serial.print(SLOW_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    Serial.print(",");
    Serial.print("SET_Flag: ");
    Serial.print(set_flag);
    Serial.println("");
    Serial.print("V_SET: ");
    Serial.println(V_SET);
  }

//  float V_BATCH_SET[TOTAL_ROWS][TOTAL_COLS];
//  for (int row = 0; row < TOTAL_ROWS; row++) 
//  {
//    for (int column = 0; column < TOTAL_COLS; column++) 
//    {
//      while (Serial.available() == 0) {}
//      V_BATCH_SET[row][column] = Serial.parseFloat();
//    }
//  }
//
//  if (Serial_println_ON == 1)
//  {
//    for (int row = 0; row < TOTAL_ROWS; row = row + 1)
//    {
//      for (int column = 0; column < TOTAL_COLS - 1; column = column + 1)
//      {
//        Serial.print(V_BATCH_SET[row][column]);
//        Serial.print(", ");
//      }
//      Serial.println(V_BATCH_SET[row][TOTAL_COLS - 1]);
//    }
//  }

//  float V_BATCH_SET_GATE[TOTAL_ROWS][TOTAL_COLS];

  for (int row = 0; row < TOTAL_ROWS; row++) 
  {
    for (int column = 0; column < TOTAL_COLS; column++) 
    {
      while (Serial.available() == 0) {}
      V_BATCH_SET_GATE[row][column] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1)
  {
    for (int row = 0; row < TOTAL_ROWS; row = row + 1)
    {
      for (int column = 0; column < TOTAL_COLS - 1; column = column + 1)
      {
        Serial.print(V_BATCH_SET_GATE[row][column]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[row][TOTAL_COLS - 1]);
    }
    return -1;
  }

  // Step 1:
  // loop setup all existing column boards:
  // Column MUX configuration table:
  // States:     Float Float Float Float Ground Slow_pulse Read Fast_pulse
  // COL_MUX_A0:   0     1     0     1      0       1        0      1
  // COL_MUX_A1:   0     0     1     1      0       0        1      1
  // COL_MUX_EN:   0     0     0     0      1       1        1      1
  for (int col_board = 0; col_board < NUM_COL_BOARDS; col_board++)
  {
    CB_BATCH_NUMBER = col_board;
    select_column_board(CB_BATCH_NUMBER);

    //Set-up Column MUX Enable********************************************
    select_latch(0);//MUX Enable Lines
    PORTE = 0; //
    pulse_the_latch();

    //Set-up Column MUX_A0 and MUX_A1********************************************
    select_latch(1);//MUX A0 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(2);//MUX A1 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(3);//Set Hi/Low to Low
    PORTE = 8;
    pulse_the_latch();

    //No need to setup TIAs since it is in SET mode********************************
    //setup_col_pulse_width(COL_BATCH_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);    
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);
    setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
  }

  // Step 2:
  // loop setup all existing row boards:
  // Row MUX configuration table:
  // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
  // ROW_MUX_A0:   0     1     0     1      0       1        0      1
  // ROW_MUX_A1:   0     0     1     1      0       0        1      1
  // ROW_MUX_EN:   0     0     0     0      1       1        1      1
  for (int row_board = 0; row_board < NUM_ROW_BOARDS; row_board++)
  {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = row_board;
    select_row_board(RB_BATCH_NUMBER);

    //set up ROW_MUX_EN_0to7
    select_latch(0);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    //set up ROW_MUX_EN_8to15
    select_latch(4);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(1);//Row A0 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(2);//Row A1 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(5);//Row A0 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(6);//Row A1 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(3);//Set Hi/Low to Low
    PORTE = 8;
    pulse_the_latch();

    //setup_row_pulse_width(COL_BATCH_PULSE_WIDTH, RB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_row_dac_span(COL_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
  }

  // Begin batch set:
  for (int row = 0; row < TOTAL_ROWS; row++)
  { 
    for (int column = 0; column < TOTAL_COLS; column++)
    { 
      digitalWrite(TRIG, HIGH);
      int current_row = row;
      int current_rb_number = row / 16;
      int current_row_in_rb = current_row % 16;

      int current_col = column;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

      if ( V_BATCH_SET_GATE[current_row][current_col] > 0 && set_flag == 1)
      {
        for (int row_board = 0; row_board < NUM_ROW_BOARDS; row_board++)
        {
          for (int i = 0; i < 8; i++)  voltage_temp[i] = V_SET;
          setup_row_dac_span(COL_DAC_BATCH_SPAN, row_board);
          setup_row0to7_dac_voltage(voltage_temp, row_board, COL_DAC_BATCH_SPAN);
          setup_row8to15_dac_voltage(voltage_temp, row_board, COL_DAC_BATCH_SPAN);
          select_row_board(row_board);        
          select_latch(0); //set up ROW_MUX_EN_0to7
          PORTE = 255;
          pulse_the_latch();
          select_latch(4); //set up ROW_MUX_EN_8to15
          PORTE = 255;
          pulse_the_latch();
        }

        select_row_board(current_rb_number);
        if (current_row_in_rb < 8)
        {
          select_latch(1);//Row A0 Address Lines
          PORTE = ~ PORTE_conversion_for_single_channel(current_row_in_rb);
          pulse_the_latch();

          select_latch(2);//Row A1 Address Lines
          PORTE = ~ PORTE_conversion_for_single_channel(current_row_in_rb);
          pulse_the_latch();
        }
        else
        {
          select_latch(5);//Row A0 Address Lines
          PORTE = ~ PORTE_conversion_for_single_channel(current_row_in_rb - 8);
          pulse_the_latch();

          select_latch(6);//Row A1 Address Lines
          PORTE = ~ PORTE_conversion_for_single_channel(current_row_in_rb - 8);
          pulse_the_latch();
        }

        // setup the gate voltage for the selected column.
        select_column_board(current_cb_number);
        select_latch(0);//MUX EN Address Lines
        PORTE = PORTE_conversion_for_single_channel(current_col_in_cb);
        pulse_the_latch();

        // Use DAC input as the slow pulse trigger.
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
        setup_col_dac_voltage_oneline(V_SET, current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);

        digitalWrite(SLOW_PULSE, HIGH);//Send Pulse That Starts the Entire Process
        delayMicroseconds(SLOW_PULSE_WIDTH);
        digitalWrite(SLOW_PULSE, LOW);

//        Serial.print(V_BATCH_SET_GATE[current_row][current_col]);
//        Serial.print(',');

        select_column_board(current_cb_number);
        select_latch(0);//MUX EN Address Lines
        PORTE = 0;
        pulse_the_latch();

        for (int row_board = 0; row_board < NUM_ROW_BOARDS; row_board++)
        {
          select_row_board(row_board);
          select_latch(0); //set up ROW_MUX_EN_0to7
          PORTE = 0;
          pulse_the_latch();
          select_latch(4); //set up ROW_MUX_EN_8to15
          PORTE = 0;
          pulse_the_latch();
        }

        select_row_board(current_rb_number);
        if (current_row_in_rb < 8)
        {
          select_latch(1);//Row A0 Address Lines
          PORTE = 255;
          pulse_the_latch();

          select_latch(2);//Row A1 Address Lines
          PORTE = 255;
          pulse_the_latch();
        }
        else
        {
          select_latch(5);//Row A0 Address Lines
          PORTE = 255;
          pulse_the_latch();

          select_latch(6);//Row A1 Address Lines
          PORTE = 255;
          pulse_the_latch();
        }
      }
      digitalWrite(TRIG, LOW);
    }
//    Serial.println();
  }

//  Serial.println("\nCleaning up");
  
  safeStartAllRowsAndColumns();

  Serial.println("cmd32 batch_set_row_voltage complete");
  return 1;
}


int batch_set_float(){ //case 34

    int MY_COUNTER0 = 0;
  int MY_COUNTER1 = 0;
  int MY_COUNTER2 = 0;

  // Read in batch reset configurations - begin.
  if (Serial_println_ON == 1) {
    Serial.println("Batch reset slow command 1: number of column boards(1~8), number of row boards(1~8), slow column pulse width, COL_DAC_BATCH_SPAN(5 or 10), V_SET");
  }
  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  while (Serial.available() == 0) {}
  SLOW_PULSE_WIDTH = Serial.parseInt();
  SLOW_PULSE_WIDTH = constrain(SLOW_PULSE_WIDTH, 10, 1000000);
  while (Serial.available() == 0) {}
  COL_DAC_BATCH_SPAN = Serial.parseInt();
  while (Serial.available() == 0) {}
  float V_SET = Serial.parseFloat();

  TOTAL_COLS = 8 * NUM_COL_BOARDS;
  TOTAL_ROWS = 16 * NUM_ROW_BOARDS;

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    Serial.print("NUM_COL_BOARDS: ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(",");
    Serial.print("NUM_ROW_BOARDS: ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(",");
    Serial.print("TOTAL_COLS: ");
    Serial.print(TOTAL_COLS);
    Serial.print(",") ;
    Serial.print("TOTAL_ROWS: ");
    Serial.print(TOTAL_ROWS);
    Serial.print(",");
    Serial.print("SLOW_PULSE_WIDTH: ");
    Serial.print(SLOW_PULSE_WIDTH);
    Serial.print(",");
    Serial.print("COL_DAC_BATCH_SPAN: ");
    Serial.print(COL_DAC_BATCH_SPAN);
    Serial.println(",");
    Serial.print("V_SET: ");
    Serial.println(V_SET);
  }

//  if (Serial_println_ON == 1) {
//    Serial.println("Batch reset command slow 2: V_BATCH_RESET for column boards. Array size is previously defined by number of column and row boards.");
//  }
//  
//  for (MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
//    for (MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
//      while (Serial.available() == 0) {}
//      // To save memory, use V_BATCH_SET array for both set and reset operations.
//      V_BATCH_SET[MY_COUNTER1][MY_COUNTER0] = V_SET;
//    }
//  }
//
//  // Verify if parameters are correctly received.
//  if (Serial_println_ON == 1) {
//    for (MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
//      for (MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
//        Serial.print(V_BATCH_SET[MY_COUNTER1][MY_COUNTER0]);
//        Serial.print(", ");
//      }
//      Serial.println(V_BATCH_SET[MY_COUNTER1][TOTAL_COLS - 1]);
//    }
//    Serial.println("All data print out.");
//  }

  if (Serial_println_ON == 1) {
    Serial.println("Batch reset command 3: V_BATCH_RESET_GATE for column boards. Array size is previously defined by number of column and row boards.");
  }

  for (MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
    for (MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS; MY_COUNTER0 = MY_COUNTER0 + 1) {
      while (Serial.available() == 0) {}
      // To save memory, use V_BATCH_SET_GATE array for both set and reset operations.
      V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0] = Serial.parseFloat();
    }
  }

  // Verify if parameters are correctly received.
  if (Serial_println_ON == 1) {
    for (MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_ROWS; MY_COUNTER1 = MY_COUNTER1 + 1) {
      for (MY_COUNTER0 = 0; MY_COUNTER0 < TOTAL_COLS - 1; MY_COUNTER0 = MY_COUNTER0 + 1) {
        Serial.print(V_BATCH_SET_GATE[MY_COUNTER1][MY_COUNTER0]);
        Serial.print(", ");
      }
      Serial.println(V_BATCH_SET_GATE[MY_COUNTER1][TOTAL_COLS - 1]);
    }
    Serial.println("All data print out.");
  }

  // Read in Batch reset configurations - complete.

  // Step 1:
  // loop setup all existing column boards:
  if (Serial_println_ON == 1) {
    Serial.println("Start loop setup all existing column boards");
  }
  for (MY_COUNTER2 = 0; MY_COUNTER2 < NUM_COL_BOARDS; MY_COUNTER2 = MY_COUNTER2 + 1) {
    CB_BATCH_NUMBER = MY_COUNTER2;
    select_column_board(CB_BATCH_NUMBER);

    // Grounding all columns: COL_MUX = 255, COL_MUX_A0 = 0, COL_MUX_A1 = 0;
    // Floating all columns while they are all in fast pulse mode: 0, 255, 255
    //Set-up Column MUX Enable********************************************
    select_latch(0);//MUX Enable Lines
    PORTE = 0; //
    pulse_the_latch();

    //Set-up Column MUX_A0 and MUX_A1********************************************
    select_latch(1);//MUX A0 Address Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(2);//MUX A1 Address Lines
    PORTE = 0;
    pulse_the_latch();

    //No need to setup TIAs since it is in reset mode********************************
    setup_col_pulse_width(COL_BATCH_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);
  }

  if (Serial_println_ON == 1) {
    Serial.println("Complete Loop setup all existing column boards");
  }
  // Step 2:
  // loop setup all existing row boards:
  for (MY_COUNTER2 = 0; MY_COUNTER2 < NUM_ROW_BOARDS; MY_COUNTER2++) {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = MY_COUNTER2;
    select_row_board(RB_BATCH_NUMBER);

    // Row MUX configuration table:
    // States:     Float Float Float Float Ground Slow_pulse Float Fast_pulse
    // ROW_MUX_A0:   0     1     0     1      0       1        0      1
    // ROW_MUX_A1:   0     0     1     1      0       0        1      1
    // ROW_MUX_EN:   0     0     0     0      1       1        1      1

    // Floating all rows while all in grounding mode.
    //set up ROW_MUX_EN_0to7
    select_latch(0);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(1);//Row A0 Address Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(2);//Row A1 Address Lines
    PORTE = 0;
    pulse_the_latch();

    //set up ROW_MUX_EN_8to15
    select_latch(4);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(5);//Row A0 Address Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(6);//Row A1 Address Lines
    PORTE = 0;
    pulse_the_latch();

    //setup_row_pulse_width(1024, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    //setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    //setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  if (Serial_println_ON == 1) {
    Serial.println("Loop setup all existing row boards");
  }
  // setup all column boards and row boards before batch reset - end.

  // Begin batch reset:
  for (MY_COUNTER2 = 0; MY_COUNTER2 < TOTAL_ROWS; MY_COUNTER2++) { // column-wise loop
    for (MY_COUNTER1 = 0; MY_COUNTER1 < TOTAL_COLS; MY_COUNTER1++) { // row-wise loop

      int current_row = MY_COUNTER2;
      int current_row_in_rb = current_row % 16;

      int current_rb_number = current_row / 16;


      int current_col = MY_COUNTER1;
      int current_col_in_cb = current_col % 8;
      int current_cb_number = current_col / 8;

      if (V_BATCH_SET_GATE[current_row][current_col] != 0) {

        select_row_board(current_rb_number);
        if (current_row_in_rb < 8) {
          // Change to the selected row in the first 8 rows:
          select_latch(0);//ROW_MUX_EN Address Lines
          PORTE = PORTE_conversion_for_single_channel(current_row_in_rb);
          pulse_the_latch();
        }
        else {
          // Change to the selected row in the last 8 rows:
          select_latch(4);//ROW_MUX_EN Address Lines
          PORTE = PORTE_conversion_for_single_channel(current_row_in_rb - 8);
          pulse_the_latch();
        }

        // setup the gate voltage for the selected column.
        select_column_board(current_cb_number);

        //Enable the selected column
        //Set-up Column MUX_A0 and MUX_A1********************************************
        //select_column_board(current_cb_number);

        select_latch(1);//Col A0 Address Lines
        PORTE = PORTE_conversion_for_single_channel(current_col_in_cb);
        pulse_the_latch();

        select_latch(0);//MUX EN Address Lines
        PORTE = PORTE_conversion_for_single_channel(current_col_in_cb);
        pulse_the_latch();

        // Use DAC input as the slow pulse trigger.
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(V_BATCH_SET_GATE[current_row][current_col], current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(V_SET, current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);

        //digitalWrite(SLOW_PULSE, HIGH);//Send Pulse That Starts the Entire Process
        delayMicroseconds(SLOW_PULSE_WIDTH);
        //digitalWrite(SLOW_PULSE, LOW);

        setup_col_dac_span(COL_DAC_BATCH_SPAN, current_cb_number);
        setup_col_dac_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_DAC_BATCH_SPAN);
        setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
        setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);

        select_latch(1);//Col A0 Address Lines
        PORTE = 0;
        pulse_the_latch();

        select_latch(0);//MUX EN Address Lines
        PORTE = 0;
        pulse_the_latch();

        // disable the selected column
        //select_column_board(current_cb_number);

        select_row_board(current_rb_number);

        // Change to the selected row in the first 8 rows:
        select_latch(0);//ROW_MUX_EN Address Lines
        PORTE = 0;
        pulse_the_latch();

        // Change to the selected row in the last 8 rows:
        select_latch(4);//ROW_MUX_EN Address Lines
        PORTE = 0;
        pulse_the_latch();
      }
    }
  }

  safeStartAllRowsAndColumns();

  Serial.println("cmd34 gate-controlled batch set (floating unselected rows) complete");
  return 1;
}
void batch_read_bin()
{
//  uint16_t temp;
  
  while (Serial.available() == 0) {}
  V_BATCH_READ = Serial.parseFloat();

  while (Serial.available() == 0) {}
  V_BATCH_GATE = Serial.parseFloat();

  while (Serial.available() == 0) {}
  TIA_BATCH_GAIN = Serial.parseInt();

  while (Serial.available() == 0) {}
  ROW_BATCH_PULSE_WIDTH = Serial.parseInt();

  while (Serial.available() == 0) {}
  COL_PULSE_WIDTH = Serial.parseInt();

  while (Serial.available() == 0) {}
  COL_BATCH_SH_DELAY = Serial.parseInt();

  while (Serial.available() == 0) {}
  COL_BATCH_AD_CONVST_DELAY = Serial.parseInt();

  while (Serial.available() == 0) {}
  NUM_COL_BOARDS = Serial.parseInt();

  while (Serial.available() == 0) {}
  NUM_ROW_BOARDS = Serial.parseInt();
  
  int TOTAL_COLS = 8 * NUM_COL_BOARDS;
  int TOTAL_ROWS = 16 * NUM_ROW_BOARDS;

  if (Serial_println_ON == 1) 
  {
    Serial.print(V_BATCH_READ);
    Serial.print(", ");
    Serial.print(V_BATCH_GATE);
    Serial.print(", ");
    Serial.print(TIA_BATCH_GAIN);
    Serial.print(", ");
    Serial.print(ROW_BATCH_PULSE_WIDTH);
    Serial.print(", ");
    Serial.print(COL_PULSE_WIDTH);
    Serial.print(", ");
    Serial.print(COL_BATCH_SH_DELAY);
    Serial.print(", ");
    Serial.print(COL_BATCH_AD_CONVST_DELAY);
    Serial.print(", ");
    Serial.print(NUM_COL_BOARDS);
    Serial.print(", ");
    Serial.print(NUM_ROW_BOARDS);
    Serial.print(", ");
    Serial.print(TOTAL_COLS);
    Serial.print(", ");
    Serial.print(TOTAL_ROWS);
    Serial.println("");
  }

  TIA_Gain_Convert(TIA_BATCH_GAIN);

  // loop setup all existing column boards:
  for (int col_board = 0; col_board < NUM_COL_BOARDS; col_board++) {
    CB_BATCH_NUMBER = col_board;
    select_column_board(CB_BATCH_NUMBER);

    select_latch(0);//MUX Enable Lines
    PORTE = 0; //
    pulse_the_latch();

    select_latch(1);//MUX A0 Address Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(2);//MUX A1 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(4);//TIA A0 Address Lines
    PORTE = TIA_BATCH_READ_A0;
    pulse_the_latch();

    select_latch(5);//TIA A1 Address Lines
    PORTE = TIA_BATCH_READ_A1;
    pulse_the_latch();

    //COL_DAC_BATCH_SPAN = 5;
    setup_col_pulse_width(COL_PULSE_WIDTH, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sample_and_hold(COL_BATCH_SH_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_adc_convst(COL_BATCH_AD_CONVST_DELAY, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_dac_span(COL_DAC_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_DAC_BATCH_SPAN);
    setup_col_sel_span(COL_SEL_BATCH_SPAN, CB_BATCH_NUMBER);
    setup_col_sel_voltage(COL_SEL_BATCH_VOLTAGE_ZERO, CB_BATCH_NUMBER, COL_SEL_BATCH_SPAN);
  }

  // loop setup all existing row boards:
  for (int row_board = 0; row_board < NUM_ROW_BOARDS; row_board++) 
  {
    // Set-up all row boards - begin ***************************************************
    //Set up MUX on Row Board ********************************************************
    RB_BATCH_NUMBER = row_board;
    select_row_board(RB_BATCH_NUMBER);

    select_latch(0);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(1);//Row A0 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(2);//Row A1 Address Lines
    PORTE = 255;
    pulse_the_latch();

    //set up ROW_MUX_EN_8to15
    select_latch(4);//Row Enable Lines
    PORTE = 0;
    pulse_the_latch();

    select_latch(5);//Row A0 Address Lines
    PORTE = 255;
    pulse_the_latch();

    select_latch(6);//Row A1 Address Lines
    PORTE = 255;
    pulse_the_latch();
    for (int i = 0; i < 8; i++) 
    {
      ROW_DAC_BATCH_VOLTAGE_0to7[i] = V_BATCH_READ;
      ROW_DAC_BATCH_VOLTAGE_8to15[i] = V_BATCH_READ;
    }
    setup_row_pulse_width(ROW_BATCH_PULSE_WIDTH, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    setup_row_dac_span(ROW_DAC_BATCH_SPAN, RB_BATCH_NUMBER);
    setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
    setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15, RB_BATCH_NUMBER, ROW_DAC_BATCH_SPAN);
  }
  // setup all column boards and row boards before batch read - end.

  // Begin batch read:
  for (int column = 0; column < TOTAL_COLS; column++) 
  { // column-wise loop
    int current_col = column;
    int current_col_in_cb = current_col % 8;
    int current_cb_number = current_col / 8;
    
    select_column_board(current_cb_number);

    select_latch(0);//MUX EN Address Lines
    PORTE = PORTE_conversion_for_single_channel(current_col_in_cb);
    pulse_the_latch();

    // Turn on the selected transistor line.
    setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
    setup_col_sel_voltage_oneline(V_BATCH_GATE, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);

    for (int row = 0; row < TOTAL_ROWS; row++) 
    { // row-wise loop
      int current_row = row;
      int current_rb_number = current_row / 16;
      int current_row_in_rb = current_row % 16;

      select_row_board(current_rb_number);
      if (current_row_in_rb < 8) 
      {
        select_latch(0);//ROW_MUX_EN Address Lines
        PORTE = PORTE_conversion_for_single_channel(current_row_in_rb);
        pulse_the_latch();
      }
      else 
      {
        select_latch(4);//ROW_MUX_EN Address Lines
        PORTE = PORTE_conversion_for_single_channel(current_row_in_rb - 8);
        pulse_the_latch();
      }

      // Do a fast read as in option 4
      select_column_board(current_cb_number);
      select_latch(6);
      PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
      pulse_the_latch();

      delayMicroseconds(1);
      digitalWrite(MPU_PULSE, HIGH);//Send Pulse That Starts the Entire Process
      delayMicroseconds(5);
      digitalWrite(MPU_PULSE, LOW);

      select_column_board(current_cb_number);
      delayMicroseconds(1);
      PORTE = 5;//Lower ADC Chip Select

      digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
      for (int i = 0; i < 8; i++) 
      {
        PORTE = 5;//Raise ADC_NOT_READ
        PORTE = 1;//Lower ADC_NOT_READ
        delayMicroseconds(1);
        ADC_READ_VALUE[i] = PORTD;
      }
      PORTE = 5;//Raise ADC_NOT_READ
      PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
      digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches

//      temp = (uint16_t)ADC_READ_VALUE[current_col_in_cb];
////      Serial.flush();
//      Serial.write((char *)&temp, 2);
      
      Serial.print(ADC_READ_VALUE[current_col_in_cb]);
      
      if( row == TOTAL_ROWS - 1) Serial.println("");
      else Serial.print(",");

      select_row_board(current_rb_number);
      select_latch(0);//ROW_MUX_EN Address Lines
      PORTE = 0;
      pulse_the_latch();
      select_latch(4);//ROW_MUX_EN Address Lines
      PORTE = 0;
      pulse_the_latch();
    }
    // change the selected column back to ground if current_col_in_cb == 7.
    if (current_col_in_cb == 7) 
    {
      select_column_board(current_cb_number);
      select_latch(0);//COL_MUX_EN Address Lines
      PORTE = 0;
      pulse_the_latch();
    }
    setup_col_sel_span(COL_SEL_BATCH_SPAN, current_cb_number);
    setup_col_sel_voltage_oneline(0, current_col_in_cb, current_cb_number, COL_SEL_BATCH_SPAN);
  }

  safeStartAllRowsAndColumns();
  Serial.println("end");
}
int batch_conv2d() // Case 40, 2D convolution
{
	// Constants =============================================================
	float v_max = 0.2; // Peak signal amplitude applied
	float voltage_divider = 630;  // The transfered data is int, divided by this number to get voltages.

	// Parameters  =============================================================

	float GATE_VOLTAGE_ONOFF[2]; // DPE gate voltage for selected/unselected columns

	int INPUT_DIM[4]; // Dimension of input tensor (rows*cols*depths*batch_size)
	int KERNEL_SIZE[4]; // Dimension of kernel (rows*cols*depths*num_kernels)
	int OUTPUT_DIM[4]; // Dimension of output tensor (rows*cols*num_kernels*batch_size)
	int STRIDES[2]; // Strides of kernels (row direction stride, col direction stride)
	float BIAS_CONFIG1; // bias_config(1) in MATLAB, bias voltage amplitude
	int BIAS_CONFIG2; // bias_config(2) in MATLAB, no. of bias inputs

	int DP_REP[2]; // Vertical differential pair repeats (row/column-wise repeats)

	// Receive circuit parameters  =============================================================

	while (Serial.available() == 0) {} // Wait for nonzero input buffer
	ROW_PULSE_WIDTH = Serial.parseInt(); // Row pulse width

	while (Serial.available() == 0) {}
	ROW_DAC_SPAN = Serial.parseInt(); // DAC range

	while (Serial.available() == 0) {}
	TIA_BATCH_GAIN = Serial.parseInt(); // TIA Gain select

	while (Serial.available() == 0) {}
	GATE_VOLTAGE_ONOFF[0] = Serial.parseFloat(); // Gate voltage for the selected columns
	GATE_VOLTAGE_ONOFF[1] = Serial.parseFloat(); // Gate voltage for the unselected columns

	// Receive convolution parameters  =============================================================

	while (Serial.available() == 0) {}
	BIAS_CONFIG1 = Serial.parseFloat(); // Bias amplitude

	while (Serial.available() == 0) {}
	BIAS_CONFIG2 = Serial.parseInt(); // No. of bias inputs

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		INPUT_DIM[i] = Serial.parseInt(); //  Input (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		KERNEL_SIZE[i] = Serial.parseInt(); //  Kernel (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		OUTPUT_DIM[i] = Serial.parseInt(); //  Output (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 2; i++)
	{
		STRIDES[i] = Serial.parseInt(); //  Strides (rows, columns) 2D
	}

	int INPUT_SIZE = INPUT_DIM[0] * INPUT_DIM[1] * INPUT_DIM[2] * INPUT_DIM[3];
	char APPLIED_VOLTAGES[INPUT_SIZE]; // Bias not included here

	while (Serial.available() == 0) {}

	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES, 1); //dump the existing 0x80
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES, INPUT_SIZE); //read the entire input

	// Receive array parameters =============================================================

	while (Serial.available() == 0) {}
	for (int i = 0; i < 2; i++)
	{
		DP_REP[i] = Serial.parseInt(); // DP repeats row/col-wise
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 16; i++)
	{
		ROW_ENABLES[i] = Serial.parseInt(); // All selected rows (should cover differntial pairs)
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 8; i++)
	{
		COL_ENABLES[i] = Serial.parseInt(); // All selected columns (should cover differntial pairs)
	}

	int ROW_ARRAY_SIZE_SINGLE = (KERNEL_SIZE[0] * KERNEL_SIZE[1] * KERNEL_SIZE[2] + BIAS_CONFIG2) * 2; // No. of rows of the entity to be repeated (with vertical DPs)
	int ROW_ARRAY_SIZE = ROW_ARRAY_SIZE_SINGLE * DP_REP[0]; // No. of total physical rows
	int COL_ARRAY_SIZE = KERNEL_SIZE[3] * DP_REP[1]; // No. of total physical columns	

	int ROW_ENABLES_MAP[ROW_ARRAY_SIZE]; // Records the picked rows, e.g. Row 2, 4, 7... (They may be not continous to bypass defect rows)
	int COL_ENABLES_MAP[COL_ARRAY_SIZE]; // Records the picked cols, e.g. Col 2, 4, 7...

	while (Serial.available() == 0) {}
	for (int i = 0; i < ROW_ARRAY_SIZE; i++)
	{
		ROW_ENABLES_MAP[i] = Serial.parseInt(); // Selected rows
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < COL_ARRAY_SIZE; i++)
	{
		COL_ENABLES_MAP[i] = Serial.parseInt(); // Selected columns
	}

	// Print received commanding  =============================================================

	if (Serial_println_ON == 1) // In case to print out the parameters
	{

		Serial.print("ROW_PULSE_WIDTH: ");
		Serial.println(ROW_PULSE_WIDTH);

		Serial.print("ROW_DAC_SPAN: ");
		Serial.println(ROW_DAC_SPAN);

		Serial.print("TIA_BATCH_GAIN: ");
		Serial.println(TIA_BATCH_GAIN);

		Serial.print("GATE_VOLTAGES: ");
		Serial.print(GATE_VOLTAGE_ONOFF[0]);
		Serial.print(",");
		Serial.println(GATE_VOLTAGE_ONOFF[1]);

		Serial.print("BIAS_VOLTAGE: ");
		Serial.println(BIAS_CONFIG1);

		Serial.print("BIAS_NUMBER: ");
		Serial.println(BIAS_CONFIG2);

		Serial.print("INPUT_DIM: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(INPUT_DIM[i]);
			Serial.print(",");
		}
		Serial.println(INPUT_DIM[3]);

		Serial.print("KERNEL_SIZE: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(KERNEL_SIZE[i]);
			Serial.print(",");
		}
		Serial.println(KERNEL_SIZE[3]);

		Serial.print("OUTPUT_DIM: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(OUTPUT_DIM[i]);
			Serial.print(",");
		}
		Serial.println(OUTPUT_DIM[3]);

		Serial.print("STRIDES: ");
		Serial.print(STRIDES[0]);
		Serial.print(",");
		Serial.println(STRIDES[1]);

		Serial.print("DP_REPEATS: ");
		Serial.print(DP_REP[0]);
		Serial.print(",");
		Serial.println(DP_REP[1]);

		Serial.print("ROW_ENABLES: ");
		for (int i = 0; i < 15; i++)
		{
			Serial.print(ROW_ENABLES[i], BIN);
			Serial.print(",");
		}
		Serial.println(ROW_ENABLES[15]);

		Serial.print("COL_ENABLES: ");
		for (int i = 0; i < 7; i++)
		{
			Serial.print(COL_ENABLES[i], BIN);
			Serial.print(",");
		}
		Serial.println(COL_ENABLES[7]);

		Serial.print("ROW_ENABLES_MAP: ");
		for (int i = 0; i < (ROW_ARRAY_SIZE - 1); i++)
		{
			Serial.print(ROW_ENABLES_MAP[i]);
			Serial.print(",");
		}
		Serial.println(ROW_ENABLES_MAP[ROW_ARRAY_SIZE - 1]);

		Serial.print("COL_ENABLES_MAP: ");
		for (int i = 0; i < (COL_ARRAY_SIZE - 1); i++)
		{
			Serial.print(COL_ENABLES_MAP[i]);
			Serial.print(",");
		}
		Serial.println(COL_ENABLES_MAP[COL_ARRAY_SIZE - 1]);

		Serial.println("APPLIED_VOLTAGES: ");
		for (int i = 0; i < INPUT_SIZE - 1; i++) // repeater for all samples in the batch
		{
			Serial.print((float)APPLIED_VOLTAGES[i] / voltage_divider, 4);
			Serial.print(",");
		}
		Serial.println((float)APPLIED_VOLTAGES[INPUT_SIZE - 1] / voltage_divider, 4);

	}

	// Hardware Initialization  =============================================================

	TIA_Gain_Convert(TIA_BATCH_GAIN); // Set TIA gain

	for (int ROW_BOARD_NUMBER = 0; ROW_BOARD_NUMBER < 8; ROW_BOARD_NUMBER++) // Row boards initializatoin
	{

		select_row_board(ROW_BOARD_NUMBER); //
		select_latch(0);
		PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2]; //
		pulse_the_latch();
		select_latch(1); // Row A0 Address Lines
		PORTE = 255;
		pulse_the_latch();
		select_latch(2); // Row A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		select_latch(4);
		PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2 + 1];
		pulse_the_latch();
		select_latch(5); //Row A0 Address Lines
		PORTE = 255;
		pulse_the_latch();
		select_latch(6); //Row A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		setup_row_pulse_width(ROW_PULSE_WIDTH, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_row_dac_span(ROW_DAC_SPAN, ROW_BOARD_NUMBER);
		setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
	}

	for (int COL_BOARD_NUMBER = 0; COL_BOARD_NUMBER < 8; COL_BOARD_NUMBER++) // Column boards initialization
	{
		select_column_board(COL_BOARD_NUMBER);
		select_latch(0); //MUX Enable Lines
		//PORTE = COL_ENABLES[COL_BOARD_NUMBER]; // Scheme 1: Enable picked columns 
		PORTE = 255; // Scheme 2: Enable all columns
		pulse_the_latch();

		select_latch(1); //MUX A0 Address Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(2); //MUX A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		select_latch(4); //TIA A0 Address Lines
		PORTE = TIA_BATCH_READ_A0;
		pulse_the_latch();

		select_latch(5); //TIA A1 Address Lines
		PORTE = TIA_BATCH_READ_A1;
		pulse_the_latch();

		for (int i = 0; i < 8; i++) // Assign V_Gate
		{
			//if (bitRead(COL_ENABLES[COL_BOARD_NUMBER], i) == 1) // Scheme 1: Enable picked column gates
			//{
			//	GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[0]; // Column in use
			//}
			//else
			//{
			//	GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[1]; // Column not in use
			//}

			GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[0]; // Scheme 2: Enable all columns
		}

		setup_col_pulse_width(ROW_PULSE_WIDTH + 20, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_sample_and_hold(ROW_PULSE_WIDTH - 100, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_adc_convst(ROW_PULSE_WIDTH + 10, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_dac_span(ROW_DAC_SPAN, COL_BOARD_NUMBER);
		setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_sel_span(COL_SEL_SPAN, COL_BOARD_NUMBER);
		setup_col_sel_voltage(GATE_VOLTAGE_TEMP, COL_BOARD_NUMBER, COL_SEL_SPAN);
	}

	// Convolution  =============================================================

	int INPUT_DIM01 = INPUT_DIM[0] * INPUT_DIM[1];
	int INPUT_DIM02 = INPUT_DIM[0] * INPUT_DIM[1] * INPUT_DIM[2];
	//int KERNEL_SIZE01 = KERNEL_SIZE[0] * KERNEL_SIZE[1];
	//int KERNEL_SIZE02 = KERNEL_SIZE[0] * KERNEL_SIZE[1] * KERNEL_SIZE[2];

	for (int sample_ID = 0; sample_ID < INPUT_DIM[3]; sample_ID++) // Go through all samples of the batch
	{
		for (int col_output = 0; col_output < OUTPUT_DIM[1]; col_output++) // Go through all columns of the output tensor
		{
			for (int row_output = 0; row_output < OUTPUT_DIM[0]; row_output++) // Go through all rows of the output tensor
			{
				// Select the "square" input 3D volume, then unroll it (into 1D) =========================================

				float voltage[ROW_ARRAY_SIZE_SINGLE];
				float max_voltage = abs(BIAS_CONFIG1); // The min max_voltage is the bias
				int current_row = 0;

				for (int dep_kernel = 0; dep_kernel < KERNEL_SIZE[2]; dep_kernel++)
				{
					for (int col_kernel = 0; col_kernel < KERNEL_SIZE[1]; col_kernel++)
					{
						for (int row_kernel = 0; row_kernel < KERNEL_SIZE[0]; row_kernel++)
						{
							int row_input = STRIDES[0] * row_output + row_kernel; // Row number in the input "square"
							int col_input = STRIDES[1] * col_output + col_kernel; // Column number in the input "square"

							float v_temp = (float)APPLIED_VOLTAGES[sample_ID * INPUT_DIM02 + dep_kernel * INPUT_DIM01 + col_input * INPUT_DIM[0] + row_input] / voltage_divider;

							// Update maximum absolute voltage
							max_voltage = max(max_voltage, abs(v_temp));

							// Record the row voltages with vertical DP pairs
							// Note current_row = (dep_kernel * KERNEL_SIZE01 + col_kernel * KERNEL_SIZE[0] + row_kernel) * 2;
							voltage[current_row] = v_temp; // Positive input row voltage
							voltage[current_row + 1] = -v_temp; // Negative input row voltage
							current_row = current_row + 2; // Update the current row by 2 (DP)							
						}
					}
				}

				for (int i = 0; i < BIAS_CONFIG2; i++) // Extra rows for bias
				{
					voltage[current_row] = BIAS_CONFIG1;
					voltage[current_row + 1] = -BIAS_CONFIG1;
					current_row = current_row + 2;
				}

				// Set voltages =========================================================================================

				if (max_voltage < 0.001)
				{
					max_voltage = v_max;
				}


				for (int row_array = 0; row_array < ROW_ARRAY_SIZE_SINGLE; row_array++) // Loop on all Vrow
				{
					for (int dp_rep0 = 0; dp_rep0 < DP_REP[0]; dp_rep0++) // Loop to apply voltages to all vertically repeating blocks
					{
						// Get current physical row number => row board number, row in row board
						int current_rb_number = (ROW_ENABLES_MAP[row_array] + dp_rep0 * ROW_ARRAY_SIZE_SINGLE) / 16;
						int current_row_in_rb = (ROW_ENABLES_MAP[row_array] + dp_rep0 * ROW_ARRAY_SIZE_SINGLE) % 16;
						
						// With dynamic scaling
						setup_row_dac_voltage_oneline(voltage[row_array] / max_voltage * v_max, current_row_in_rb, current_rb_number, ROW_DAC_SPAN);
						// Without dynamic scaling
						//setup_row0to7_dac_voltage_oneline(voltage[row_array], current_row_in_rb, current_rb_number, ROW_DAC_SPAN);                
					}
				}

				// Send voltages =========================================================================================

				digitalWrite(MPU_PULSE, HIGH); // Send Pulse That Starts the Entire Process
				delayMicroseconds(5); // Manual timer
				digitalWrite(MPU_PULSE, LOW); //

				// Receive data on columns ==============================================================================

				//float scaling_back = max_voltage / v_max;

				for (int CB_NUMBER = 0; CB_NUMBER < 8; CB_NUMBER++) // !!! Note that all boards are used.
				{
					select_column_board(CB_NUMBER);
					select_latch(6); //Preselect latch with ADC control lines
					PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
					pulse_the_latch();
					PORTE = 5;//Lower ADC Not Chip Select
					digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
					for (int i = 0; i < 8; i++)
					{
						PORTE = 5;//Raise ADC_NOT_READ
						PORTE = 1;//Lower ADC_NOT_READ
						delayMicroseconds(1);
						ADC_READ_VALUE[i] = PORTD; // Why use array???
					
						if (bitRead(COL_ENABLES[CB_NUMBER], i) == 1) // Send outputs of selected columns (output every column even with horizontal differential pairs)
						{							
							int16_t temp = (int16_t)(ADC_READ_VALUE[i]);
							Serial.write((char *)&temp, 2); // Send back data without using memory
						}						
					}
					PORTE = 5;//Raise ADC_NOT_READ
					PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
					digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches
				}
			}
		}
		Serial.println("end");
	}
	return 1;
}
int batch_conv2dlstm() // Case 41, 2D convolution lstm
{
	// Constants =============================================================
	float v_max = 0.2; // Peak signal amplitude applied
	float voltage_divider = 630;  // The transfered data is int, divided by this number to get voltages.

	// Parameters  =============================================================

	float GATE_VOLTAGE_ONOFF[2]; // DPE gate voltage for selected/unselected columns

	int INPUT_DIM_X[4]; // Dimension of input tensor (rows*cols*depths*batch_size)
	int INPUT_DIM_H[4]; // Dimension of recurrent input tensor (rows*cols*depths*batch_size)
	int KERNEL_SIZE_X[4]; // Dimension of kernel (rows*cols*depths*num_kernels)
	int KERNEL_SIZE_H[4]; // Dimension of recurrent kernel (rows*cols*depths*num_kernels)
	int OUTPUT_DIM[4]; // Dimension of output tensor (rows*cols*num_kernels*batch_size)
	int STRIDES_X[2]; // Strides of input kernels (row direction stride, col direction stride)
	float BIAS_CONFIG1; // bias_config(1) in MATLAB, bias voltage amplitude
	int BIAS_CONFIG2; // bias_config(2) in MATLAB, no. of bias inputs

	int DP_REP[2]; // Vertical differential pair repeats (row/column-wise repeats)

	// Receive circuit parameters  =============================================================

	while (Serial.available() == 0) {} // Wait for nonzero input buffer
	ROW_PULSE_WIDTH = Serial.parseInt(); // Row pulse width

	while (Serial.available() == 0) {}
	ROW_DAC_SPAN = Serial.parseInt(); // DAC range

	while (Serial.available() == 0) {}
	TIA_BATCH_GAIN = Serial.parseInt(); // TIA Gain select

	while (Serial.available() == 0) {}
	GATE_VOLTAGE_ONOFF[0] = Serial.parseFloat(); // Gate voltage for the selected columns
	GATE_VOLTAGE_ONOFF[1] = Serial.parseFloat(); // Gate voltage for the unselected columns

	// Receive convolution parameters  =============================================================

	while (Serial.available() == 0) {}
	BIAS_CONFIG1 = Serial.parseFloat(); // Bias amplitude

	while (Serial.available() == 0) {}
	BIAS_CONFIG2 = Serial.parseInt(); // No. of bias inputs

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		INPUT_DIM_X[i] = Serial.parseInt(); //  Input (4D tensor) dimension 
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		INPUT_DIM_H[i] = Serial.parseInt(); //  Recurrent input (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		KERNEL_SIZE_X[i] = Serial.parseInt(); //  Kernel (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		KERNEL_SIZE_H[i] = Serial.parseInt(); //  Recurrent kernel (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 4; i++)
	{
		OUTPUT_DIM[i] = Serial.parseInt(); //  Output (4D tensor) dimension
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 2; i++)
	{
		STRIDES_X[i] = Serial.parseInt(); //  Input strides (rows, columns) 2D
	}

	int INPUT_SIZE_X = INPUT_DIM_X[0] * INPUT_DIM_X[1] * INPUT_DIM_X[2] * INPUT_DIM_X[3]; // Input
	int INPUT_SIZE_H = INPUT_DIM_H[0] * INPUT_DIM_H[1] * INPUT_DIM_H[2] * INPUT_DIM_H[3]; // Recurrent input
	char APPLIED_VOLTAGES_X[INPUT_SIZE_X]; // Bias not included here
	char APPLIED_VOLTAGES_H[INPUT_SIZE_H]; // Bias not included here
	while (Serial.available() == 0) {}
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES_X, 1); //dump the existing 0x80
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES_X, INPUT_SIZE_X); //read the entire input
	Serial.readBytesUntil(0x80, APPLIED_VOLTAGES_H, INPUT_SIZE_H); //read the entire recurrent input

	// Receive array parameters =============================================================

	while (Serial.available() == 0) {}
	for (int i = 0; i < 2; i++)
	{
		DP_REP[i] = Serial.parseInt(); // DP repeats row/col-wise
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 16; i++)
	{
		ROW_ENABLES[i] = Serial.parseInt(); // All selected rows (should cover differntial pairs)
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < 8; i++)
	{
		COL_ENABLES[i] = Serial.parseInt(); // All selected columns (should cover differntial pairs)
	}

	int ROW_ARRAY_SIZE_SINGLE = (KERNEL_SIZE_X[0] * KERNEL_SIZE_X[1] * KERNEL_SIZE_X[2] + KERNEL_SIZE_H[0] * KERNEL_SIZE_H[1] * KERNEL_SIZE_H[2] + BIAS_CONFIG2) * 2; // No. of rows of the entity to be repeated (with vertical DPs)
	int ROW_ARRAY_SIZE = ROW_ARRAY_SIZE_SINGLE * DP_REP[0];  // No. of total physical rows
	int COL_ARRAY_SIZE = KERNEL_SIZE_X[3] * 4 * DP_REP[1];  // No. of total physical columns	

	int ROW_ENABLES_MAP[ROW_ARRAY_SIZE];  // Records the picked rows, e.g. Row 2, 4, 7... (They may be not continous to bypass defect rows)
	int COL_ENABLES_MAP[COL_ARRAY_SIZE];  // Records the picked cols, e.g. Col 2, 4, 7... Note KERNEL_SIZE_X[3] should == KERNEL_SIZE_H[3]

	while (Serial.available() == 0) {}
	for (int i = 0; i < ROW_ARRAY_SIZE; i++)
	{
		ROW_ENABLES_MAP[i] = Serial.parseInt(); // Selected rows
	}

	while (Serial.available() == 0) {}
	for (int i = 0; i < COL_ARRAY_SIZE; i++)
	{
		COL_ENABLES_MAP[i] = Serial.parseInt(); // Selected columns
	}

	// Print received commanding  =============================================================

	if (Serial_println_ON == 1) // In case to print out the parameters
	{

		Serial.print("ROW_PULSE_WIDTH: ");
		Serial.println(ROW_PULSE_WIDTH);

		Serial.print("ROW_DAC_SPAN: ");
		Serial.println(ROW_DAC_SPAN);

		Serial.print("TIA_BATCH_GAIN: ");
		Serial.println(TIA_BATCH_GAIN);

		Serial.print("GATE_VOLTAGES: ");
		Serial.print(GATE_VOLTAGE_ONOFF[0]);
		Serial.print(",");
		Serial.println(GATE_VOLTAGE_ONOFF[1]);

		Serial.print("BIAS_VOLTAGE: ");
		Serial.println(BIAS_CONFIG1);

		Serial.print("BIAS_NUMBER: ");
		Serial.println(BIAS_CONFIG2);

		Serial.print("INPUT_DIM_X: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(INPUT_DIM_X[i]);
			Serial.print(",");
		}
		Serial.println(INPUT_DIM_X[3]);

		Serial.print("INPUT_DIM_H: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(INPUT_DIM_H[i]);
			Serial.print(",");
		}
		Serial.println(INPUT_DIM_H[3]);

		Serial.print("KERNEL_SIZE_X: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(KERNEL_SIZE_X[i]);
			Serial.print(",");
		}
		Serial.println(KERNEL_SIZE_X[3]);

		Serial.print("KERNEL_SIZE_H: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(KERNEL_SIZE_H[i]);
			Serial.print(",");
		}
		Serial.println(KERNEL_SIZE_H[3]);

		Serial.print("OUTPUT_DIM: ");
		for (int i = 0; i < 3; i++)
		{
			Serial.print(OUTPUT_DIM[i]);
			Serial.print(",");
		}
		Serial.println(OUTPUT_DIM[3]);

		Serial.print("STRIDES_X: ");
		Serial.print(STRIDES_X[0]);
		Serial.print(",");
		Serial.println(STRIDES_X[1]);

		Serial.print("DP_REPEATS: ");
		Serial.print(DP_REP[0]);
		Serial.print(",");
		Serial.println(DP_REP[1]);

		Serial.print("ROW_ENABLES: ");
		for (int i = 0; i < 15; i++)
		{
			Serial.print(ROW_ENABLES[i], BIN);
			Serial.print(",");
		}
		Serial.println(ROW_ENABLES[15]);

		Serial.print("COL_ENABLES: ");
		for (int i = 0; i < 7; i++)
		{
			Serial.print(COL_ENABLES[i], BIN);
			Serial.print(",");
		}
		Serial.println(COL_ENABLES[7]);

		Serial.print("ROW_ENABLES_MAP: ");
		for (int i = 0; i < (ROW_ARRAY_SIZE - 1); i++)
		{
			Serial.print(ROW_ENABLES_MAP[i]);
			Serial.print(",");
		}
		Serial.println(ROW_ENABLES_MAP[ROW_ARRAY_SIZE - 1]);

		Serial.print("COL_ENABLES_MAP: ");
		for (int i = 0; i < (COL_ARRAY_SIZE - 1); i++)
		{
			Serial.print(COL_ENABLES_MAP[i]);
			Serial.print(",");
		}
		Serial.println(COL_ENABLES_MAP[COL_ARRAY_SIZE - 1]);

		Serial.println("APPLIED_VOLTAGES_X: ");
		for (int i = 0; i < INPUT_SIZE_X - 1; i++) // repeater for all samples in the batch
		{
			Serial.print((float)APPLIED_VOLTAGES_X[i] / voltage_divider, 4);
			Serial.print(",");
		}
		Serial.println((float)APPLIED_VOLTAGES_X[INPUT_SIZE_X - 1] / voltage_divider, 4);

		Serial.println("APPLIED_VOLTAGES_H: ");
		for (int i = 0; i < INPUT_SIZE_H - 1; i++) // repeater for all samples in the batch
		{
			Serial.print((float)APPLIED_VOLTAGES_H[i] / voltage_divider, 4);
			Serial.print(",");
		}
		Serial.println((float)APPLIED_VOLTAGES_H[INPUT_SIZE_H - 1] / voltage_divider, 4);

	}

	// Hardware Initialization  =============================================================

	TIA_Gain_Convert(TIA_BATCH_GAIN); // Set TIA gain

	for (int ROW_BOARD_NUMBER = 0; ROW_BOARD_NUMBER < 8; ROW_BOARD_NUMBER++) // Row boards initializatoin
	{

		select_row_board(ROW_BOARD_NUMBER); //
		select_latch(0);
		PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2]; //
		pulse_the_latch();
		select_latch(1); // Row A0 Address Lines
		PORTE = 255;
		pulse_the_latch();
		select_latch(2); // Row A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		select_latch(4);
		PORTE = ROW_ENABLES[ROW_BOARD_NUMBER * 2 + 1];
		pulse_the_latch();
		select_latch(5); //Row A0 Address Lines
		PORTE = 255;
		pulse_the_latch();
		select_latch(6); //Row A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		setup_row_pulse_width(ROW_PULSE_WIDTH, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_row_dac_span(ROW_DAC_SPAN, ROW_BOARD_NUMBER);
		setup_row0to7_dac_voltage(ROW_DAC_BATCH_VOLTAGE_0to7_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_row8to15_dac_voltage(ROW_DAC_BATCH_VOLTAGE_8to15_ZERO, ROW_BOARD_NUMBER, ROW_DAC_SPAN);
	}

	for (int COL_BOARD_NUMBER = 0; COL_BOARD_NUMBER < 8; COL_BOARD_NUMBER++) // Column boards initialization
	{
		select_column_board(COL_BOARD_NUMBER);
		select_latch(0); //MUX Enable Lines
		//PORTE = COL_ENABLES[COL_BOARD_NUMBER]; // Scheme 1: Enable picked columns 
		PORTE = 255; // Scheme 2: Enable all columns
		pulse_the_latch();

		select_latch(1); //MUX A0 Address Lines
		PORTE = 0;
		pulse_the_latch();

		select_latch(2); //MUX A1 Address Lines
		PORTE = 255;
		pulse_the_latch();

		select_latch(4); //TIA A0 Address Lines
		PORTE = TIA_BATCH_READ_A0;
		pulse_the_latch();

		select_latch(5); //TIA A1 Address Lines
		PORTE = TIA_BATCH_READ_A1;
		pulse_the_latch();

		for (int i = 0; i < 8; i++) // Assign V_Gate
		{
			//if (bitRead(COL_ENABLES[COL_BOARD_NUMBER], i) == 1) // Scheme 1: Enable picked column gates
			//{
			//	GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[0]; // Column in use
			//}
			//else
			//{
			//	GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[1]; // Column not in use
			//}

			GATE_VOLTAGE_TEMP[i] = GATE_VOLTAGE_ONOFF[0]; // Scheme 2: Enable all columns
		}

		setup_col_pulse_width(ROW_PULSE_WIDTH + 20, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_sample_and_hold(ROW_PULSE_WIDTH - 100, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_adc_convst(ROW_PULSE_WIDTH + 10, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_dac_span(ROW_DAC_SPAN, COL_BOARD_NUMBER);
		setup_col_dac_voltage(COL_DAC_BATCH_VOLTAGE_ZERO, COL_BOARD_NUMBER, ROW_DAC_SPAN);
		setup_col_sel_span(COL_SEL_SPAN, COL_BOARD_NUMBER);
		setup_col_sel_voltage(GATE_VOLTAGE_TEMP, COL_BOARD_NUMBER, COL_SEL_SPAN);
	}

	// Convolution  =============================================================

	int INPUT_DIM_X01 = INPUT_DIM_X[0] * INPUT_DIM_X[1];
	int INPUT_DIM_X02 = INPUT_DIM_X[0] * INPUT_DIM_X[1] * INPUT_DIM_X[2];
	int INPUT_DIM_H01 = INPUT_DIM_H[0] * INPUT_DIM_H[1];
	int INPUT_DIM_H02 = INPUT_DIM_H[0] * INPUT_DIM_H[1] * INPUT_DIM_H[2];
	//int KERNEL_SIZE_X01 = KERNEL_SIZE_X[0] * KERNEL_SIZE_X[1];
	//int KERNEL_SIZE_X02 = KERNEL_SIZE_X[0] * KERNEL_SIZE_X[1] * KERNEL_SIZE_X[2];
	//int KERNEL_SIZE_H01 = KERNEL_SIZE_H[0] * KERNEL_SIZE_H[1];
	//int KERNEL_SIZE_H02 = KERNEL_SIZE_H[0] * KERNEL_SIZE_H[1] * KERNEL_SIZE_H[2];

	for (int sample_ID = 0; sample_ID < INPUT_DIM_X[3]; sample_ID++) // Go through all samples of the batch
	{
		for (int col_output = 0; col_output < OUTPUT_DIM[1]; col_output++) // Go through all columns of the output tensor
		{
			for (int row_output = 0; row_output < OUTPUT_DIM[0]; row_output++) // Go through all rows of the output tensor
			{

				// Select the "square" volume of the input tensor, and unroll it, make it input to the chip ==============

				float voltage[ROW_ARRAY_SIZE_SINGLE];
				float max_voltage = abs(BIAS_CONFIG1); // The min max_voltage is the bias
				int current_row = 0;

				for (int dep_kernel = 0; dep_kernel < KERNEL_SIZE_X[2]; dep_kernel++)
				{
					for (int col_kernel = 0; col_kernel < KERNEL_SIZE_X[1]; col_kernel++)
					{
						for (int row_kernel = 0; row_kernel < KERNEL_SIZE_X[0]; row_kernel++)
						{
							int row_input = STRIDES_X[0] * row_output + row_kernel; // Row number in the input "square"
							int col_input = STRIDES_X[1] * col_output + col_kernel; // Column number in the input "square"

							float v_temp = (float)APPLIED_VOLTAGES_X[sample_ID * INPUT_DIM_X02 + dep_kernel * INPUT_DIM_X01 + col_input * INPUT_DIM_X[0] + row_input] / voltage_divider;

							// Update maximum absolute voltage
							max_voltage = max(max_voltage, abs(v_temp));
							// Record the row voltages with vertical DP pairs
							// Note current_row = (dep_kernel * KERNEL_SIZE_X01 + col_kernel * KERNEL_SIZE_X[0] + row_kernel) * 2;
							voltage[current_row] = v_temp; // Positive input row voltage
							voltage[current_row + 1] = -v_temp; // Negative input row voltage
							current_row = current_row + 2; // Update the current row by 2 (DP)

						}
					}
				}

				// Select the "square" volume of the recurrent input tensor, and unroll it, make it input to the chip ====

				for (int dep_kernel = 0; dep_kernel < KERNEL_SIZE_H[2]; dep_kernel++)
				{
					for (int col_kernel = 0; col_kernel < KERNEL_SIZE_H[1]; col_kernel++)
					{
						for (int row_kernel = 0; row_kernel < KERNEL_SIZE_H[0]; row_kernel++)
						{
							int row_input = row_output + row_kernel; // Row number in the input "square", default stride = 1
							int col_input = col_output + col_kernel; // Column number in the input "square", , default stride = 1

							float v_temp = (float)APPLIED_VOLTAGES_H[sample_ID * INPUT_DIM_H02 + dep_kernel * INPUT_DIM_H01 + col_input * INPUT_DIM_H[0] + row_input] / voltage_divider;

							// Update maximum absolute voltage
							max_voltage = max(max_voltage, abs(v_temp));

							// Record the row voltages with vertical DP pairs
							// Note current_row = (KERNEL_SIZE_X02 + dep_kernel * KERNEL_SIZE_H01 + col_kernel * KERNEL_SIZE_H[0] + row_kernel) * 2;
							voltage[current_row] = v_temp; // Positive input row voltage
							voltage[current_row + 1] = -v_temp; // Negative input row voltage
							current_row = current_row + 2; // Update the current row by 2 (DP)

						}
					}
				}

				// Bias
				for (int i = 0; i < BIAS_CONFIG2; i++) // Extra rows for bias
				{
					voltage[current_row] = BIAS_CONFIG1;
					voltage[current_row + 1] = -BIAS_CONFIG1;
					current_row = current_row + 2;
				}

				// Set voltages =========================================================================================

				if (max_voltage < 0.001)
				{
					max_voltage = v_max;
				}


				for (int row_array = 0; row_array < ROW_ARRAY_SIZE_SINGLE; row_array++) // Loop on all Vrow
				{
					for (int dp_rep0 = 0; dp_rep0 < DP_REP[0]; dp_rep0++) // Loop to apply voltages to all vertically repeating blocks
					{
						// Get current physical row number => row board number, row in row board
						int current_rb_number = (ROW_ENABLES_MAP[row_array] + dp_rep0 * ROW_ARRAY_SIZE_SINGLE) / 16;
						int current_row_in_rb = (ROW_ENABLES_MAP[row_array] + dp_rep0 * ROW_ARRAY_SIZE_SINGLE) % 16;

						// With dynamic scaling
						setup_row_dac_voltage_oneline(voltage[row_array] / max_voltage * v_max, current_row_in_rb, current_rb_number, ROW_DAC_SPAN);
						// Without dynamic scaling
						//setup_row0to7_dac_voltage_oneline(voltage[row_array], current_row_in_rb, current_rb_number, ROW_DAC_SPAN);                
					}
				}

				// Send voltages =========================================================================================

				digitalWrite(MPU_PULSE, HIGH); // Send Pulse That Starts the Entire Process
				delayMicroseconds(5); // Manual timer
				digitalWrite(MPU_PULSE, LOW); //

				// Receive data on columns ==============================================================================

				//float scaling_back = max_voltage / v_max;

				for (int CB_NUMBER = 0; CB_NUMBER < 8; CB_NUMBER++) // !!! Note that all boards are used.
				{
					select_column_board(CB_NUMBER);
					select_latch(6); //Preselect latch with ADC control lines
					PORTE = 7;//Set ADC_NOT_READ, ADC_NOT_CS, and ADC_NOT_WRITE HIGH
					pulse_the_latch();
					PORTE = 5;//Lower ADC Not Chip Select
					digitalWrite(NOT_BOARD_OE[0], LOW);//Enable writing to latches
					for (int i = 0; i < 8; i++)
					{
						PORTE = 5;//Raise ADC_NOT_READ
						PORTE = 1;//Lower ADC_NOT_READ
						delayMicroseconds(1);
						ADC_READ_VALUE[i] = PORTD; // Why use array???

						if (bitRead(COL_ENABLES[CB_NUMBER], i) == 1) // Send outputs of selected columns (output every column even with horizontal differential pairs)
						{
							int16_t temp = (int16_t)(ADC_READ_VALUE[i]);
							Serial.write((char *)&temp, 2); // Send back data without using memory
						}
					}
					PORTE = 5;//Raise ADC_NOT_READ
					PORTE = 7;//Raise ADC_NOT_CS -- Done Reading All 8 Channels
					digitalWrite(NOT_BOARD_OE[0], HIGH);//Disable writing to latches
				}
			}
		}
		Serial.println("end");
	}
	return 1;
}