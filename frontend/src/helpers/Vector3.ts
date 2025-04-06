import * as THREE from 'three'

// Type structure for a standard info type and transmition format
export type SpaceInfo={
    position:SimpleVec,
    rotation:SimpleVec,
    scale:SimpleVec
}
export type SimpleVec={x:number, y:number, z:number}

export function AddSimpVec(Vec1:SimpleVec, Vec2:SimpleVec) {
    return {
        x:Vec1.x+Vec2.x,
        y:Vec1.y+Vec2.y,
        z:Vec1.z+Vec2.z
    }
}

export function SubSimpVec(Vec1:SimpleVec, Vec2:SimpleVec) {
    return {
        x:Vec1.x-Vec2.x,
        y:Vec1.y-Vec2.y,
        z:Vec1.z-Vec2.z
    }
}

export function LerpSimpVec(Vec1:SimpleVec, Vec2:SimpleVec, MaxInterp:number=0.1) {
    return {
        x:THREE.MathUtils.lerp(Vec1.x, Vec2.x, MaxInterp),
        y:THREE.MathUtils.lerp(Vec1.y, Vec2.y, MaxInterp),
        z:THREE.MathUtils.lerp(Vec1.y, Vec2.z, MaxInterp)
    }
}

export function rotationAngle(Vec:SimpleVec) {
    return {
        x:Math.atan(Vec.y/Vec.z) || 0,
        y:Math.atan(Vec.x/Vec.z) || 0,
        z:Math.atan(Vec.y/Vec.x) || 0
    }
}

export function vecLength(Vec:SimpleVec) {
    return Math.sqrt(Math.pow(Vec.x, 2) + Math.pow(Vec.y, 2) + Math.pow(Vec.z, 2))
}

export function scalarDiv(Vec:SimpleVec, Scalar:number) {
    return {
        x:Vec.x/Scalar,
        y:Vec.y/Scalar,
        z:Vec.z/Scalar
    }
}