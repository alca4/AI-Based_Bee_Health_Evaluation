import { useRef } from 'react';
import { StyleSheet } from 'react-native';
import Video, { VideoRef } from 'react-native-video';

// Within your render function, assuming you have a file called
// "background.mp4" in your project. You can include multiple videos
// on a single screen if you like.

export const VideoPlayer = (props: any) => {
  const videoRef = useRef<VideoRef>(null);
  const { uri } = props;
  let background = uri ? { uri } : require('../../uploads/VID20240924150329.mp4');
  // background = require('../../uploads/VID20240924150329.mp4');
  // //  useRef

  console.log("preVideo",uri);

  //  <StyleSheet
  return (
    <Video
      // Can be a URL or a local file.
      source={background}
      // Store reference  
      ref={videoRef}
      repeat={true}                   // make it a loop
      paused={false}                  // make it start    

      // Callback when remote video is buffering                                      
      onBuffer={() => {

      }}
      // Callback when video cannot be loaded              
      onError={(e:any) => {
        console.log("e",e)
      }}
      style={styles.backgroundVideo}
    />
  )
}

// Later on in your styles..
var styles = StyleSheet.create({
  backgroundVideo: {
    // position: 'absolute',
    width:300,
    height:200,
    top: 0,
    left: 0,
    bottom: 0,
    right: 0,
  },
});
