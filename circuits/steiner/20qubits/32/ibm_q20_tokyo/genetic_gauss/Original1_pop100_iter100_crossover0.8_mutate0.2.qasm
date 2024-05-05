// Initial wiring: [17, 5, 9, 15, 12, 8, 10, 2, 13, 14, 3, 1, 4, 16, 7, 11, 19, 0, 6, 18]
// Resulting wiring: [17, 5, 9, 15, 12, 8, 10, 2, 13, 14, 3, 1, 4, 16, 7, 11, 19, 0, 6, 18]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[4], q[2];
cx q[10], q[6];
cx q[6], q[0];
cx q[12], q[0];
cx q[14], q[13];
cx q[16], q[12];
cx q[16], q[11];
cx q[17], q[11];
cx q[15], q[3];
cx q[12], q[10];
cx q[18], q[19];
cx q[13], q[15];
cx q[11], q[17];
cx q[7], q[17];
cx q[7], q[15];
cx q[15], q[7];
cx q[6], q[9];
cx q[9], q[13];
cx q[3], q[17];
cx q[3], q[11];
cx q[3], q[5];
cx q[2], q[14];
cx q[2], q[11];
cx q[2], q[5];
cx q[1], q[3];
cx q[1], q[2];
cx q[0], q[5];
cx q[0], q[2];
cx q[3], q[10];
cx q[0], q[9];
cx q[5], q[8];
cx q[3], q[6];
