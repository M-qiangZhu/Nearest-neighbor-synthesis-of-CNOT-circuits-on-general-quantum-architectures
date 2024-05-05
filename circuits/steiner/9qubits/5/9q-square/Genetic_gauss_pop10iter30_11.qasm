// Initial wiring: [0 1 2 4 3 5 7 6 8]
// Resulting wiring: [0 1 2 7 8 5 4 6 3]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[8], q[3];
cx q[3], q[2];
cx q[4], q[5];
cx q[3], q[8];
cx q[3], q[8];
cx q[3], q[8];
cx q[4], q[7];
cx q[4], q[7];
cx q[4], q[7];
cx q[1], q[4];
cx q[2], q[3];
