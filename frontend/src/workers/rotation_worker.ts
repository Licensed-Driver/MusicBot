import * as THREE from 'three'


let globalTimer = 0

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

function LerpSimpVec(Vec1:SimpleVec, Vec2:SimpleVec, MaxInterp:number=0.1) {
    return {
        x:THREE.MathUtils.lerp(Vec1.x, Vec2.x, MaxInterp),
        y:THREE.MathUtils.lerp(Vec1.y, Vec2.y, MaxInterp),
        z:THREE.MathUtils.lerp(Vec1.y, Vec2.z, MaxInterp)
    }
}

let albumInfo:SpaceInfo = {position:{x:0, y:0, z:0}, scale:{x:1, y:1, z:1}, rotation:{x:0, y:0, z:0}}
const albumScaleBig:SimpleVec = {x:1.25, y:1.25, z:1}
const albumScaleReg:SimpleVec = {x:1, y:1, z:1}

self.onmessage = (e) => {

    const {objectID, type, objectInfo, refPosition}:{objectID:string, type:string, objectInfo:SpaceInfo, refPosition:SimpleVec} = e.data

    switch(objectID) {
        case 'ALBUM':
            if(type=='HOVER') {
                // If the album is hovered over it'll expand over time for a smooth interaction
                albumInfo.scale = LerpSimpVec(albumInfo.scale, albumScaleBig, 0.02)
            } else if(type=='UNHOVER' && ((objectInfo.scale.x !== 1) || (objectInfo.scale.y !== 1))) {
                albumInfo.scale = LerpSimpVec(albumInfo.scale, albumScaleReg, 0.02)
            }
            if(type=='LOOK-AT') {
                let vecToRef = SubSimpVec(refPosition, objectInfo.position)
                vecToRef = LerpSimpVec(albumInfo.rotation, vecToRef, 0.1)
                albumInfo.rotation = {x:vecToRef.x, y:vecToRef.y, z:0}
            }
            if(type=='BOUNCE') {
                albumInfo.position = {x:0, y:Math.sin(globalTimer), z:0}
            }
            self.postMessage({objectID:'ALBUM', objectInfo:albumInfo})
    }
        
}

const timer = () => {
    globalTimer += 0.016
}

setInterval(timer, 16)