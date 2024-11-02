import React, { useEffect, useState } from 'react';
import ContentView from '../layouts/auth/sign-in-1';
import { StyleSheet, View, ScrollViewProps, Image, PermissionsAndroid } from 'react-native';
import { Button, Input, Text } from '@ui-kitten/components';
import { dailyForecast, showWeather, getLocation } from 'react-native-weather-api';
import { ImageOverlay } from '../components/image-overlay.component';
import ForecastSearch from '../components/ForecastSearch';
import moment from "moment";
import { get } from 'lodash';
import Geolocation from 'react-native-geolocation-service';


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


export async function getWeather(args: any) {
  let result;
  let URL;

  if (args.city != null) {
    URL =
      'https://api.openweathermap.org/data/2.5/weather?q=' +
      args.city +
      ',' +
      args.country +
      '&appid=' +
      args.key +
      '&units=' +
      args.unit +
      '&lang=' +
      args.lang;
  } else if (args.zip_code != null) {
    URL =
      'https://api.openweathermap.org/data/2.5/weather?zip=' +
      args.zip_code +
      ',' +
      args.country +
      '&appid=' +
      args.key +
      '&units=' +
      args.unit +
      '&lang=' +
      args.lang;
  } else {
    URL =
      'https://api.openweathermap.org/data/2.5/weather?lat=' +
      args.lat +
      '&lon=' +
      args.lon +
      '&appid=' +
      args.key +
      '&units=' +
      args.unit +
      '&lang=' +
      args.lang;
  }

  await fetch(URL)
    .then(res => res.json())
    .then(data => {
      // current = data;
      result = Promise.resolve(data);
    });

  return result;
}


function kelvinToFahrenheit(kelvin: any) {
  return ((kelvin - 273.15) * 9 / 5 + 32).toFixed(2);
}


export const HomeScreen = ({ navigation }): React.ReactElement => {

  const [email, setEmail] = React.useState<string>();
  const [password, setPassword] = React.useState<string>();

  // useState

  const [toggleSearch, setToggleSearch] = useState("city");
  const [city, setCity] = useState("New York");
  const [postalCode, setPostalCode] = useState("L4W1S9");
  const [lat, setLat] = useState(43.6532);
  const [long, setLong] = useState(-79.3832);
  const [weather, setWeather] = useState({});
  const [key, setKey] = useState("8c3961265858245573a35b4a6894b178");

  const controller = new AbortController();
  const signal = controller.signal;


  const requestLocationPermission = async () => {
    try {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        {
          title: 'Cool Photo App Location Permission',
          message:
            'Cool Photo App needs access to your location ' +
            'so you can take awesome pictures.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        },
      );
      if (granted === PermissionsAndroid.RESULTS.GRANTED) {
        console.log('You can use the location');
      } else {
        console.log('Location permission denied');
      }
    } catch (err) {
      console.warn(err);
    }
  };

  const getLocation = () => {
    // Geolocation.getCurrentPosition(
    //   position => {
    //     console.log("position", position);
    //     setLat(position.coords.latitude);
    //     setLong(position.coords.longitude);
    //   },
    //   error => {
    //     // See error code charts below.
    //     console.log(error.code, error.message);
    //   },
    //   { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 },
    // );


  };

  useEffect(() => {
    const requestLocationPermission = async () => {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION, {
          'title': 'Location Access Required',
          'message': 'This App needs to Access your location'
        }
        )
        if (granted === PermissionsAndroid.RESULTS.GRANTED) {
          //To Check, If Permission is granted
          // setGrantedPermission(true)
          // this._getCurrentLocation();
          console.log("????");
          // Geolocation.getCurrentPosition(
          //   (position) => {
          //     console.log(position);
          //   },
          //   (error) => {
          //     // See error code charts below.
          //     console.log(error.code, error.message);
          //   },
          //   { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
          // );
        } else {
          // alert("Permission Denied");
        }
      } catch (err) {
        // alert("err",err);
      }
    }
    requestLocationPermission();
  }, [])


  useEffect(() => {

    let temp;
    let wind;

    requestLocationPermission();

    // getLocation();

    getWeather({

      key: "8c3961265858245573a35b4a6894b178",
      // city: city,
      city,
      country: "US"

    }).then((res: any) => {

      console.log("showWeather", res);
      setWeather(res)

    });

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

  const wdata = {
    "base": "stations", "clouds": { "all": 100 }, "cod": 200, "coord": { "lat": 51.5085, "lon": -0.1257 }, "dt": 1729392730, "id": 2643743, "main":
      { "feels_like": 284.15, "grnd_level": 1011, "humidity": 92, "pressure": 1014, "sea_level": 1014, "temp": 284.55, "temp_max": 285.42, "temp_min": 283.15 }, "name": "London", "sys": { "country": "GB", "id": 2075535, "sunrise": 1729406002, "sunset": 1729443406, "type": 2 }, "timezone": 3600, "visibility": 10000, "weather": [{ "description": "overcast clouds", "icon": "04n", "id": 804, "main": "Clouds" }], "wind": { "deg": 160, "speed": 2.57 }
  }


  const weatherIcon = get(weather, "weather.0.icon", null)
  const weatherMain = get(weather, "weather.0.main", null)
  const weatherDescription = get(weather, "weather.0.description", null)
  const weatherMore = get(weather, "main", null)


  return (
    <KeyboardAvoidingView>


      <View style={styles.baseContainer}>







        <View style={styles.formContainer}>

          <Input
            style={styles.passwordInput}
            // secureTextEntry={true}
            placeholder='city'
            label=''
            status='control'
            value={city}
            onChangeText={setCity}
          />

          <Button
            status='control'
            // size='large'
            onPress={() => {

              getWeather({

                key: "8c3961265858245573a35b4a6894b178",
                // city: city,
                city,
                country: "US"

              }).then((res: any) => {

                console.log("showWeather", res);
                setWeather(res)

              });

            }}>
            Search
          </Button>
          <View style={{
            margin: 10,
            marginTop: 70
          }}>


            {weatherIcon && <View>
              <Text style={styles.text}><Text style={styles.text2}>Current Weather</Text></Text>
              <Text style={styles.text}><Text style={styles.text2}>City:</Text>{city}</Text>
              <Image source={{ uri: `https://openweathermap.org/img/w/${weatherIcon}.png` }} resizeMode="cover"
                resizeMethod="scale"
                style={styles.image} />
              <Text style={styles.text}>
                <Text style={styles.text2}>Status:</Text>  {weatherMain}</Text>
              <Text style={styles.text}><Text style={styles.text2}>Description:</Text>  {weatherDescription}</Text>

              {weatherMore && <View>

                {/* <Text style={styles.text}>feels_like:{weatherMore.feels_like}</Text> */}
                <Text style={styles.text}><Text style={styles.text2}>Ground level:</Text>  {weatherMore.grnd_level}</Text>
                <Text style={styles.text}><Text style={styles.text2}>Humidity:</Text>  {weatherMore.humidity}</Text>
                {/* <Text style={styles.text}>pressure:{weatherMore.pressure}</Text> */}
                <Text style={styles.text}><Text style={styles.text2}>Sea Level:</Text>  {weatherMore.sea_level}</Text>

                <Text style={styles.text}><Text style={styles.text2}>Temperature:</Text>  {kelvinToFahrenheit(weatherMore.temp)}</Text>
                <Text style={styles.text}><Text style={styles.text2}>Temperature (Max):</Text>  {kelvinToFahrenheit(weatherMore.temp_max)}</Text>
                <Text style={styles.text}><Text style={styles.text2}>Temperature (Min):</Text>  {kelvinToFahrenheit(weatherMore.temp_min)}</Text>
              </View>}





            </View>}

          </View>



        </View>

        <View style={styles.socialAuthContainer}>

        </View>
      </View>
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
  baseContainer: {
    // marginTop: 48,
    backgroundColor: "#fce38a",
    flex: 1
  },
  text: {
    color: "black",
    fontSize:16,

  },
  text2: {
    color: "black",
    fontSize:18,
    marginRight:10,
    paddingRight:10,
    fontWeight: "900"
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
    color: "black",
    textAlign: "center",
    alignItems: "center",
    justifyContent: "center"
  },
  passwordInput: {
    marginTop: 16,
    margin: 10,
    backgroundColor: "#000"

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
