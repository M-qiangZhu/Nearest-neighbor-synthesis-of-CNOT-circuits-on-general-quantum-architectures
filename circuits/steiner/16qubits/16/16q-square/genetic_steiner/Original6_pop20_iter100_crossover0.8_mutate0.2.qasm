// Initial wiring: [8, 9, 7, 15, 5, 10, 11, 12, 0, 2, 13, 14, 4, 3, 6, 1]
// Resulting wiring: [8, 9, 7, 15, 5, 10, 11, 12, 0, 2, 13, 14, 4, 3, 6, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[7], q[0];
cx q[10], q[9];
cx q[10], q[5];
cx q[13], q[10];
cx q[10], q[5];
cx q[13], q[10];
cx q[15], q[8];
cx q[13], q[14];
cx q[10], q[11];
cx q[9], q[14];
cx q[9], q[10];
cx q[8], q[15];
cx q[6], q[9];
cx q[9], q[14];
