// Initial wiring: [2, 16, 8, 10, 6, 18, 0, 4, 15, 17, 14, 19, 9, 3, 12, 1, 13, 11, 7, 5]
// Resulting wiring: [2, 16, 8, 10, 6, 18, 0, 4, 15, 17, 14, 19, 9, 3, 12, 1, 13, 11, 7, 5]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[7], q[1];
cx q[8], q[1];
cx q[9], q[8];
cx q[8], q[7];
cx q[10], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[3];
cx q[8], q[2];
cx q[8], q[7];
cx q[11], q[8];
cx q[12], q[6];
cx q[12], q[11];
cx q[6], q[3];
cx q[11], q[9];
cx q[11], q[8];
cx q[12], q[6];
cx q[13], q[7];
cx q[13], q[6];
cx q[7], q[1];
cx q[6], q[5];
cx q[6], q[3];
cx q[15], q[13];
cx q[13], q[7];
cx q[15], q[13];
cx q[16], q[13];
cx q[18], q[12];
cx q[12], q[6];
cx q[6], q[3];
cx q[12], q[6];
cx q[18], q[12];
cx q[19], q[18];
cx q[18], q[17];
cx q[18], q[11];
cx q[19], q[18];
cx q[13], q[15];
cx q[10], q[11];
cx q[9], q[11];
cx q[8], q[9];
cx q[7], q[8];
cx q[8], q[9];
cx q[9], q[11];
cx q[9], q[8];
cx q[6], q[12];
cx q[6], q[7];
cx q[4], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[6], q[12];
cx q[8], q[11];
cx q[8], q[9];
cx q[3], q[4];
cx q[2], q[8];
cx q[1], q[8];
cx q[8], q[11];
cx q[11], q[8];
