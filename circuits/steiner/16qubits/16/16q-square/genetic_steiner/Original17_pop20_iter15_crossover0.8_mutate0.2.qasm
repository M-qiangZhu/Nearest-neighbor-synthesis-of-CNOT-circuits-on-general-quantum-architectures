// Initial wiring: [2, 6, 5, 8, 0, 4, 1, 10, 11, 14, 9, 12, 15, 13, 3, 7]
// Resulting wiring: [2, 6, 5, 8, 0, 4, 1, 10, 11, 14, 9, 12, 15, 13, 3, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1], q[0];
cx q[3], q[2];
cx q[6], q[5];
cx q[5], q[4];
cx q[5], q[2];
cx q[7], q[6];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[5], q[2];
cx q[7], q[0];
cx q[8], q[7];
cx q[11], q[4];
cx q[13], q[10];
cx q[10], q[9];
cx q[14], q[13];
cx q[13], q[14];
cx q[11], q[12];
cx q[10], q[13];
cx q[13], q[14];
cx q[14], q[13];
cx q[9], q[14];
cx q[9], q[10];
cx q[8], q[15];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[9];
cx q[4], q[5];
