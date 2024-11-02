// import React from 'react'
// import { WebView } from 'react-native-webview'

// const WebViewScreen = ({ navigation }) => {


//   return (
//     <WebView
//       source={{
//         uri: 'http://183.136.204.122:32264/'
//       }}
//       style={{
//         flex: 1
//       }}
//     />
//   )
// }

// export default WebViewScreen


import React, { useEffect, useState } from 'react';
import ContentView from '../layouts/auth/sign-in-1';
import { StyleSheet, View, ScrollViewProps, Image, PermissionsAndroid } from 'react-native';
import { Button, Input, Text } from '@ui-kitten/components';
import { dailyForecast, showWeather, getLocation } from 'react-native-weather-api';
import { ImageOverlay } from '../components/image-overlay.component';
import ForecastSearch from '../components/ForecastSearch';
import moment from "moment";
import { get } from 'lodash';
// import { WebView } from 'react-native-webview'
import { WebView } from 'react-native-webview';

import Geolocation from 'react-native-geolocation-service';
// import Geolocation from '@react-native-community/geolocation';


// import {MapView, Marker} from 'react-native-maps';

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




export const WebViewScreen = ({ navigation }): React.ReactElement => {

  const [email, setEmail] = React.useState<string>();
  const [password, setPassword] = React.useState<string>();

  // useState

  const [toggleSearch, setToggleSearch] = useState("city");
  const [city, setCity] = useState("London");
  const [postalCode, setPostalCode] = useState("L4W1S9");
  const [lat, setLat] = useState(43.6532);
  const [long, setLong] = useState(-79.3832);
  const [weather, setWeather] = useState({});
  const [key, setKey] = useState("8c3961265858245573a35b4a6894b178");

  const controller = new AbortController();
  const signal = controller.signal;



  useEffect(() => {


  }, [])


  // useEffect(() => {
  //   fetch(
  //     `https://api.openweathermap.org/data/2.5/onecall?lat=${lat}&lon=${long}&exclude=hourly,minutely&units=metric&appid=${key}`,
  //     { signal }
  //   )
  //     .then((res) => res.json())
  //     .then((data) => {
  //       setWeather(data);
  //       console.log("data",data);
  //     })
  //     .catch((err) => {
  //       console.log("error", err);
  //     });

  // }, [lat, long]);




  useEffect(() => {



  }, [])

  const onSignInButtonPress = (): void => {
    navigation && navigation.goBack();
  };

  const onSignUpButtonPress = (): void => {
    navigation && navigation.navigate('SignUp1');
  };

  const fetchLatLongHandler = () => {

  }

  const fetchByPostalHandler = () => {

  }

  const weatherIcon = get(weather, "weather.0.icon", null)
  const weatherMain = get(weather, "weather.0.main", null)
  const weatherDescription = get(weather, "weather.0.description", null)
  // http://183.136.204.122:32264/
  const url = "http://183.136.204.122:32264/";
  return (
    <KeyboardAvoidingView>

      <WebView
        source={{
          uri: url
        }}
        style={{
          flex: 1
        }}
      />

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
  image: {
    width: 100,
    height: 100,
    alignSelf: "center",
    alignContent: "center"
  },
  socialAuthContainer: {
    marginTop: 48,
  },
  evaButton: {
    maxWidth: 72,
    paddingHorizontal: 0,
  },
  formContainer: {
    flex: 1,
    marginTop: 48,
    textAlign: "center",
    alignItems: "center",
    justifyContent: "center"
  },
  passwordInput: {
    marginTop: 16,
  },
  signInLabel: {
    flex: 1,
  },
  signUpButton: {
    flexDirection: 'row-reverse',
    paddingHorizontal: 0,
  },
  socialAuthButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-evenly',
  },
  socialAuthHintText: {
    alignSelf: 'center',
    marginBottom: 16,
  },
});
