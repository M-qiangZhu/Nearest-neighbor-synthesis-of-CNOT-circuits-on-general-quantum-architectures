// Initial wiring: [13, 4, 2, 1, 14, 3, 0, 10, 6, 8, 15, 9, 11, 7, 5, 12]
// Resulting wiring: [13, 4, 2, 1, 14, 3, 0, 10, 6, 8, 15, 9, 11, 7, 5, 12]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[5], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[4], q[3];
cx q[5], q[4];
cx q[9], q[6];
cx q[11], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[4], q[3];
cx q[11], q[4];
cx q[12], q[3];
cx q[13], q[12];
cx q[12], q[11];
cx q[11], q[10];
cx q[12], q[11];
cx q[14], q[13];
cx q[13], q[12];
cx q[14], q[13];
cx q[15], q[0];
cx q[11], q[12];
cx q[8], q[9];
cx q[9], q[10];
cx q[3], q[4];
cx q[4], q[11];
