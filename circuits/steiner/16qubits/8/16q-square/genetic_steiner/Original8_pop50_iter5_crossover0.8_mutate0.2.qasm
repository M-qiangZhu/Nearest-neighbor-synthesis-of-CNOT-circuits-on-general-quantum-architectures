// Initial wiring: [5, 8, 10, 15, 4, 3, 6, 11, 0, 13, 9, 7, 12, 1, 14, 2]
// Resulting wiring: [5, 8, 10, 15, 4, 3, 6, 11, 0, 13, 9, 7, 12, 1, 14, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[4], q[3];
cx q[5], q[2];
cx q[2], q[1];
cx q[5], q[2];
cx q[6], q[5];
cx q[9], q[6];
cx q[11], q[4];
cx q[4], q[3];
cx q[11], q[4];
cx q[12], q[11];
cx q[13], q[12];
cx q[12], q[11];
cx q[13], q[12];
cx q[14], q[9];
cx q[15], q[14];
cx q[14], q[9];
cx q[9], q[6];
cx q[15], q[14];
cx q[11], q[12];
cx q[6], q[9];
cx q[5], q[6];
cx q[2], q[5];
cx q[5], q[6];
