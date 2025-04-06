import * as Vec3 from '../helpers/Vector3'

let titleInfo:Vec3.SpaceInfo = {position:{x:0, y:0, z:0}, scale:{x:1, y:1, z:1}, rotation:{x:0, y:0, z:0}}

self.onmessage = (e) => {

    const {objectID, type, objectInfo, refPosition}:{objectID:string, type:string, objectInfo:Vec3.SpaceInfo, refPosition:Vec3.SimpleVec} = e.data

    if(objectID=='TITLE') {
        if(type=='LOOK-AT') {
            let vecToRef = Vec3.SubSimpVec(refPosition, objectInfo.position)
            vecToRef = Vec3.rotationAngle(vecToRef)
            titleInfo.rotation = {x:-vecToRef.x, y:vecToRef.y, z:0}
        }
        self.postMessage({objectID:'TITLE', objectInfo:titleInfo})
    } 
}