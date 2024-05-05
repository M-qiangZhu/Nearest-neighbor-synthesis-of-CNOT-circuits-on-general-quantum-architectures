// Initial wiring: [2, 11, 1, 6, 5, 12, 14, 13, 7, 9, 4, 10, 0, 3, 8, 15]
// Resulting wiring: [2, 11, 1, 6, 5, 12, 14, 13, 7, 9, 4, 10, 0, 3, 8, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[9], q[8];
cx q[10], q[9];
cx q[9], q[8];
cx q[12], q[11];
cx q[13], q[10];
cx q[10], q[9];
cx q[14], q[13];
cx q[13], q[14];
cx q[10], q[13];
cx q[13], q[14];
cx q[14], q[13];
cx q[8], q[15];
cx q[6], q[9];
cx q[9], q[8];
cx q[9], q[10];
cx q[8], q[15];
cx q[10], q[13];
cx q[15], q[8];
cx q[5], q[6];
cx q[3], q[4];
cx q[0], q[1];
