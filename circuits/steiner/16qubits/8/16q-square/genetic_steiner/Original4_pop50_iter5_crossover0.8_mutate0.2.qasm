// Initial wiring: [5, 2, 4, 12, 8, 9, 1, 10, 7, 11, 0, 6, 13, 3, 14, 15]
// Resulting wiring: [5, 2, 4, 12, 8, 9, 1, 10, 7, 11, 0, 6, 13, 3, 14, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[4], q[3];
cx q[5], q[2];
cx q[7], q[6];
cx q[8], q[7];
cx q[15], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[15], q[8];
cx q[9], q[10];
cx q[4], q[5];
cx q[2], q[5];
cx q[5], q[2];
cx q[1], q[2];
cx q[2], q[5];
cx q[2], q[1];
cx q[5], q[2];
