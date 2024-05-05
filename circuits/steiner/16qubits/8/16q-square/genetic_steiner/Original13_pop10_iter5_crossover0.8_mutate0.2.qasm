// Initial wiring: [1, 13, 3, 10, 5, 11, 8, 6, 4, 7, 15, 2, 0, 12, 14, 9]
// Resulting wiring: [1, 13, 3, 10, 5, 11, 8, 6, 4, 7, 15, 2, 0, 12, 14, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7], q[6];
cx q[6], q[5];
cx q[6], q[1];
cx q[7], q[6];
cx q[8], q[7];
cx q[14], q[13];
cx q[15], q[14];
cx q[14], q[13];
cx q[14], q[15];
cx q[13], q[14];
cx q[12], q[13];
cx q[13], q[14];
cx q[14], q[15];
cx q[15], q[14];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[15];
cx q[7], q[6];
cx q[8], q[7];
cx q[5], q[6];
cx q[6], q[7];
cx q[1], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[7], q[6];
cx q[8], q[7];
