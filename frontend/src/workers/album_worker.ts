import * as Vec3 from '../helpers/Vector3'

let globalTimer = 0

let albumInfo:Vec3.SpaceInfo = {position:{x:0, y:0, z:0}, scale:{x:1, y:1, z:1}, rotation:{x:0, y:0, z:0}}
const albumScaleBig:Vec3.SimpleVec = {x:1.25, y:1.25, z:1}
const albumScaleReg:Vec3.SimpleVec = {x:1, y:1, z:1}

self.onmessage = (e) => {

    const {objectID, type, objectInfo, refPosition}:{objectID:string, type:string, objectInfo:Vec3.SpaceInfo, refPosition:Vec3.SimpleVec} = e.data

    if(objectID=='ALBUM') {
        if(type=='HOVER') {
            // If the album is hovered over it'll expand over time for a smooth interaction
            albumInfo.scale = Vec3.LerpSimpVec(albumInfo.scale, albumScaleBig, 0.02)
        } else if(type=='UNHOVER' && ((objectInfo.scale.x !== 1) || (objectInfo.scale.y !== 1))) {
            albumInfo.scale = Vec3.LerpSimpVec(albumInfo.scale, albumScaleReg, 0.02)
        }
        if(type=='LOOK-AT') {
            let vecToRef = Vec3.SubSimpVec(refPosition, objectInfo.position)
            vecToRef = Vec3.rotationAngle(vecToRef)
            albumInfo.rotation = {x:-vecToRef.x, y:vecToRef.y, z:0}
        }
        if(type=='BOUNCE') {
            albumInfo.position = {x:0, y:Math.sin(globalTimer)/2, z:0}
        }
        self.postMessage({objectID:'ALBUM', objectInfo:albumInfo})
    }
}

const timer = () => {
    globalTimer += 0.016
}

setInterval(timer, 16)