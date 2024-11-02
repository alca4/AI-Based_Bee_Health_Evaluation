import React, { useEffect, useState } from 'react';
import ContentView from '../layouts/auth/sign-in-1';
import { StyleSheet, View, ScrollViewProps, Image, Platform } from 'react-native';
import { Button, Spinner, Text, Modal, Card } from '@ui-kitten/components';
import { dailyForecast, showWeather, getLocation } from 'react-native-weather-api';
import { ImageOverlay } from '../components/image-overlay.component';
import ForecastSearch from '../components/ForecastSearch';
import moment from "moment";
import * as ImagePicker from 'react-native-image-picker';
import { BottomSheet, ListItem } from 'react-native-elements'
import { get } from 'lodash';
import { VideoPlayer } from '../components/video-player';
import { SERVER_URL } from '../App';
import {
  LineChart,
  BarChart,
  PieChart,
  ProgressChart,
  ContributionGraph,
  StackedBarChart
} from "react-native-chart-kit";

// const SERVER_URL = 'http://192.168.86.24:5000';



// import { KeyboardAvoidingView } from '../layouts/';

export const KeyboardAvoidingView = (props): React.ReactElement => {
  const lib = require('react-native-keyboard-aware-scroll-view');

  const defaultProps: ScrollViewProps = {
    style: { flex: 1 },
    contentContainerStyle: { flexGrow: 1 },
    bounces: false,
    bouncesZoom: false,
    alwaysBounceVertical: false,
    alwaysBounceHorizontal: false,
  };

  return React.createElement(lib.KeyboardAwareScrollView, {
    enableOnAndroid: true,
    ...defaultProps,
    ...props,
  });
};

const createFormData = (assets: any, body: any = {}) => {
  const data = new FormData();
  const photo = assets.assets[0];

  data.append('photo', {
    name: photo.fileName,
    type: photo.type,
    uri: Platform.OS === 'ios' ? photo.uri.replace('file://', '') : photo.uri,
  });

  Object.keys(body).forEach((key) => {
    data.append(key, body[key]);
  });

  console.log("data", data, photo);

  return data;
};





export const HealthyScreen = ({ navigation }): React.ReactElement => {

  const [preVideo, setPreVide] = React.useState<string>();

  // useState

  const [toggleSearch, setToggleSearch] = useState("city");
  const [city, setCity] = useState("London");
  const [postalCode, setPostalCode] = useState("L4W1S9");
  const [lat, setLat] = useState(43.6532);
  const [long, setLong] = useState(-79.3832);
  const [weather, setWeather] = useState({});
  const [type, setType] = useState("image");
  const [pieData, setPieData] = useState(null);

  const [response, setResponse] = React.useState<any>(null);

  const [visible, setVisible] = React.useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [step, setStep] = React.useState(0);


  const [selectedIndex, setSelectedIndex] = React.useState(null);




  const handleUploadPhoto = () => {
    const uploadUrl = `${SERVER_URL}/upload`;
    console.log("uploadUrl", uploadUrl);

    setIsLoading(true)
    setPieData(null);
    setStep(0);
    fetch(uploadUrl, {
      method: 'POST',
      body: createFormData(response, { type }),
      headers: {
        'Content-Type': 'multipart/form-data'
      },
    })
      .then((response) => response.json())
      .then((responseJson) => {
        console.log("responseJson", responseJson);
        if (responseJson.success) {
          setIsLoading(false)
          setStep(1);

        }
        if (type == 'video' || type == 'audio' || type == 'image' && responseJson.success) {
          setPreVide(`${SERVER_URL}/${responseJson.url}`)
          console.log("responseJson.url", responseJson.url);
          console.log("responseJson", responseJson.data);
          if (responseJson.data && responseJson.data.length > 0) {
            let videoData = responseJson.data[0].map((item: any, index: any) => {
              return {
                name: item,
                population: responseJson.data[1][index],
                color: `rgba(${Math.ceil(Math.random() * 250)}, ${Math.ceil(Math.random() * 250)}, ${Math.ceil(Math.random() * 250)}, 1)`,
                legendFontColor: "#7F7F7F",
                legendFontSize: 12
              };
            });
            setPieData([...videoData])
            console.log("videoData", videoData, responseJson.data[1]);
          }

        }
        // return responseJson.movies;
      })
      .catch((error) => {
        console.error(error);
        setIsLoading(false)

      });

  };

  const onItemSelect = (index): void => {
    setSelectedIndex(index);
    setVisible(false);
  };

  const list = [
    {
      title: 'Camera', onPress: () => {
        if (type == 'image') {
          ImagePicker.launchCamera({ mediaType: "photo" }, setResponse);
        } else {
          ImagePicker.launchCamera({ mediaType: "video" }, setResponse);
        }
        setIsVisible(false)
      }
    },
    {
      title: 'ImageLibrary', onPress: () => {



        if (type == 'image') {
          ImagePicker.launchImageLibrary({ mediaType: "photo" }, setResponse);

        } else {
          ImagePicker.launchImageLibrary({ mediaType: "video" }, setResponse);

        }

        setIsVisible(false)
      }
    },
    {
      title: 'Cancel',
      containerStyle: { backgroundColor: 'red' },
      titleStyle: { color: 'white' },
      onPress: () => setIsVisible(false),
    },
  ];





  const controller = new AbortController();
  const signal = controller.signal;
  const renderToggleButton = (): React.ReactElement => (
    <Button onPress={() => setVisible(true)}>
      TOGGLE MENU
    </Button>
  );

  const onSelect = (index): void => {
    setSelectedIndex(index);
    setVisible(false);
  };




  useEffect(() => {



  }, [])

  const onButtonPress = (ctype = 'video'): void => {
    setIsVisible(true);
    setType(ctype);
  };

  const onSignUpButtonPress = (): void => {
    navigation && navigation.navigate('SignUp1');
  };

  const image = get(response, "assets.0.uri", "");

  // const preVideo = get(response, "assets.0.uri", "");
  const data = [
    {
      name: "Seoul",
      population: 0,
      color: "rgba(131, 167, 234, 1)",
      legendFontColor: "#7F7F7F",
      legendFontSize: 12
    },
    {
      name: "Toronto",
      population: 0,
      color: "#F00",
      legendFontColor: "#7F7F7F",
      legendFontSize: 12
    },
    {
      name: "Beijing",
      population: 0,
      color: "red",
      legendFontColor: "#7F7F7F",
      legendFontSize: 12
    },
    {
      name: "Moscow",
      population: 100,
      color: "rgb(0, 0, 255)",
      legendFontColor: "#7F7F7F",
      legendFontSize: 12
    }
  ];

  const chartConfig = {
    backgroundGradientFrom: "#1E2923",
    backgroundGradientFromOpacity: 0,
    backgroundGradientTo: "#08130D",
    backgroundGradientToOpacity: 0.5,
    
    color: (opacity = 1) => `rgba(26, 255, 146, ${opacity})`,
    strokeWidth: 1, // optional, default 3
    barPercentage: 0.5,
    useShadowColorFromDataset: false // optional
  };


  return (
    <KeyboardAvoidingView>

      <View style={styles.baseContainer}>

        {/* <Text>{JSON.stringify({ image,preVideo })}</Text> */}
        <View style={styles.signInContainer}>

          {/* <ForecastSearch */}

          {/* <Text>{JSON.stringify(response.assets)}</Text> */}



          {type == "image" && image && <Image source={{ uri: image }} resizeMode="cover"
            resizeMethod="scale"
            style={styles.image} />}


          {type == "video" && step == 1 && preVideo && <VideoPlayer uri={preVideo} ></VideoPlayer>}

        </View>
        <Modal
          visible={isLoading}
          backdropStyle={styles.backdrop}
          onBackdropPress={() => setVisible(false)}
        >
          <Card disabled={true}>
            <Spinner />

          </Card>
        </Modal>

        <View>
        </View>

        <Button
          status='control'
          size='large'
          style={styles.socialAuthButtonsContainer}
          onPress={() => {
            onButtonPress('video')
          }}>
          VIDEO
        </Button>
        <Button
          status='control'
          size='large'
          style={styles.socialAuthButtonsContainer}
          onPress={() => {
            onButtonPress('audio')
          }}>
          AUDIO
        </Button>
        <Button
          status='control'
          size='large'
          style={styles.socialAuthButtonsContainer}
          onPress={() => {
            onButtonPress("image");
          }}>
          IMAGE
        </Button>

        {image && <Button
          status='success'
          size='large'
          style={styles.socialAuthButtonsContainer}
          onPress={() => {
            // onButtonPress("image");
            handleUploadPhoto();
          }}>
          UPLOAD
        </Button>}

        {/* {step == 1 && <Button
          status='success'
          size='large'
          style={styles.socialAuthButtonsContainer}
          onPress={() => {
            // onButtonPress("image");
            // handleUploadPhoto();
          }}>
          To MOdal
        </Button>} */}

        {/* <PieChart
          data={data}
          width={400}
          height={220}
          chartConfig={chartConfig}
          accessor={"population"}
          backgroundColor={"transparent"}
          paddingLeft={"0"}

          // hasLegend={false}
          // center={[50, 0]}
          absolute
        /> */}

        {step == 1 && pieData && <PieChart
          data={pieData}
          width={400}
          height={220}
          chartConfig={chartConfig}
          accessor={"population"}
          backgroundColor={"transparent"}
          paddingLeft={"0"}

          // hasLegend={false}
          // center={[50, 0]}
          absolute
        />}

        <BottomSheet
          isVisible={isVisible}
          containerStyle={{ backgroundColor: 'rgba(0.5, 0.25, 0, 0.2)' }}
        >
          {list.map((l, i) => (
            <ListItem key={i} containerStyle={l.containerStyle} onPress={l.onPress}>
              <ListItem.Content>
                <ListItem.Title style={l.titleStyle}>{l.title}</ListItem.Title>
              </ListItem.Content>
            </ListItem>
          ))}
        </BottomSheet>


        <View style={styles.socialAuthContainer}>
          <Text
            style={styles.socialAuthHintText}
            status='control'>
            {/* Sign with a social account */}
          </Text>
          <View style={styles.socialAuthButtonsContainer}>
            <Button
              appearance='ghost'
              size='giant'
              status='control'

            />
            <Button
              appearance='ghost'
              size='giant'
              status='control'

            />
            <Button
              appearance='ghost'
              size='giant'
              status='control'

            />
          </View>
        </View>
      </View>
      {/* </ImageOverlay> */}
    </KeyboardAvoidingView>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingVertical: 24,
    paddingHorizontal: 16,
  },
  signInContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 24,
  },
  socialAuthContainer: {
    marginTop: 48,
  },
  backdrop: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  evaButton: {
    maxWidth: 72,
    paddingHorizontal: 0,
  },
  baseContainer: {
    // marginTop: 48,
    backgroundColor: "#fce38a",
    flex: 1
  },
  text: {
    color: "black",
  },
  formContainer: {
    flex: 1,
    marginTop: 48,
  },
  passwordInput: {
    marginTop: 16,
  },
  signInLabel: {
    flex: 1,
  },
  image: {
    width: 200,
    height: 200,
  },
  signUpButton: {
    flexDirection: 'row-reverse',
    paddingHorizontal: 0,
  },
  socialAuthButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-evenly',
    marginTop: 10
  },
  socialAuthHintText: {
    alignSelf: 'center',
    marginBottom: 16,
  },
});
