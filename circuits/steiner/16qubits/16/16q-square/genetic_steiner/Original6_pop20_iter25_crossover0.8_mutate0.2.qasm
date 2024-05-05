// Initial wiring: [6, 14, 3, 10, 4, 5, 2, 13, 12, 8, 7, 15, 1, 0, 9, 11]
// Resulting wiring: [6, 14, 3, 10, 4, 5, 2, 13, 12, 8, 7, 15, 1, 0, 9, 11]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[6], q[1];
cx q[7], q[6];
cx q[6], q[5];
cx q[6], q[1];
cx q[7], q[0];
cx q[7], q[6];
cx q[8], q[7];
cx q[11], q[10];
cx q[10], q[9];
cx q[14], q[9];
cx q[15], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[7], q[0];
cx q[15], q[8];
cx q[12], q[13];
cx q[13], q[12];
cx q[8], q[9];
cx q[6], q[7];
cx q[2], q[3];
