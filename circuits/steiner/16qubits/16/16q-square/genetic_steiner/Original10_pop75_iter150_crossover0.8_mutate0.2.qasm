// Initial wiring: [6, 15, 14, 11, 10, 5, 12, 0, 9, 4, 13, 2, 1, 8, 7, 3]
// Resulting wiring: [6, 15, 14, 11, 10, 5, 12, 0, 9, 4, 13, 2, 1, 8, 7, 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4], q[3];
cx q[9], q[8];
cx q[10], q[5];
cx q[5], q[4];
cx q[10], q[9];
cx q[4], q[3];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[6];
cx q[6], q[1];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[10], q[9];
cx q[11], q[10];
cx q[13], q[10];
cx q[14], q[15];
cx q[15], q[14];
cx q[3], q[4];
cx q[4], q[11];
cx q[0], q[1];
