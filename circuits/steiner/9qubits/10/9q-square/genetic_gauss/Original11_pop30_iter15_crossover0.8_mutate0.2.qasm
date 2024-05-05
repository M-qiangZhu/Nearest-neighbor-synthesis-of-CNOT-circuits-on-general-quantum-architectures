// Initial wiring: [4, 2, 0, 1, 5, 3, 6, 7, 8]
// Resulting wiring: [4, 2, 0, 1, 5, 3, 6, 7, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[4];
cx q[5], q[1];
cx q[8], q[7];
cx q[6], q[0];
cx q[7], q[3];
cx q[6], q[7];
cx q[1], q[7];
