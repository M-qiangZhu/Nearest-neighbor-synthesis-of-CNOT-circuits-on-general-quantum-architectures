// Initial wiring: [5 1 2 4 0 3 7 6 8]
// Resulting wiring: [5 1 2 4 0 3 7 6 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[1];
cx q[4], q[1];
cx q[0], q[1];
cx q[3], q[2];
cx q[7], q[8];
