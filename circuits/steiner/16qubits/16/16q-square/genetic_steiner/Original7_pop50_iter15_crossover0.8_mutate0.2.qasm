// Initial wiring: [11, 13, 1, 14, 4, 6, 3, 9, 8, 15, 5, 12, 0, 10, 7, 2]
// Resulting wiring: [11, 13, 1, 14, 4, 6, 3, 9, 8, 15, 5, 12, 0, 10, 7, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[6], q[5];
cx q[5], q[2];
cx q[7], q[0];
cx q[8], q[7];
cx q[9], q[6];
cx q[6], q[1];
cx q[9], q[6];
cx q[10], q[9];
cx q[9], q[6];
cx q[6], q[1];
cx q[10], q[5];
cx q[13], q[10];
cx q[10], q[5];
cx q[5], q[2];
cx q[2], q[1];
cx q[10], q[5];
cx q[14], q[9];
cx q[9], q[6];
cx q[9], q[14];
cx q[9], q[10];
cx q[8], q[9];
cx q[9], q[14];
cx q[14], q[9];
cx q[7], q[8];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[9];
cx q[4], q[11];
cx q[0], q[7];
cx q[7], q[8];
cx q[8], q[7];
