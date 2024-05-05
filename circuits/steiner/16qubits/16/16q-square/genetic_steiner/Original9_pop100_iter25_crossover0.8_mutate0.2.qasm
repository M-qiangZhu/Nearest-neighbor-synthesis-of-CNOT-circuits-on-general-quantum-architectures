// Initial wiring: [9, 8, 0, 11, 5, 2, 14, 13, 10, 15, 1, 12, 7, 4, 6, 3]
// Resulting wiring: [9, 8, 0, 11, 5, 2, 14, 13, 10, 15, 1, 12, 7, 4, 6, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[8], q[7];
cx q[11], q[10];
cx q[10], q[9];
cx q[10], q[5];
cx q[9], q[8];
cx q[5], q[2];
cx q[8], q[7];
cx q[10], q[5];
cx q[12], q[11];
cx q[13], q[10];
cx q[10], q[5];
cx q[13], q[12];
cx q[5], q[2];
cx q[12], q[11];
cx q[10], q[5];
cx q[13], q[10];
cx q[13], q[12];
cx q[14], q[15];
cx q[12], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[7], q[8];
cx q[6], q[7];
cx q[3], q[4];
cx q[2], q[5];
cx q[5], q[10];
