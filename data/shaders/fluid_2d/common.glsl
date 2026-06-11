#ifdef BOUNDARY_SET
layout(set = BOUNDARY_SET, binding = 0) uniform sampler2D boundaryField;
#endif

#define st(p) fract(p)
