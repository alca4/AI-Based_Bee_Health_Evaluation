import React, { useEffect, useState } from 'react';
import ContentView from '../layouts/auth/sign-in-1';
import { StyleSheet, View, ScrollViewProps } from 'react-native';
import { Button, Input, Text } from '@ui-kitten/components';
import { dailyForecast, showWeather, getLocation } from 'react-native-weather-api';
import { ImageOverlay } from '../components/image-overlay.component';
import ForecastSearch from '../components/ForecastSearch';
import moment from "moment";
import { Divider, List, ListItem } from '@ui-kitten/components';
import { SERVER_URL } from '../App';
import { useFocusEffect } from '@react-navigation/core';
import { get } from 'lodash';

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


// interface IListItem {
//   title: string;
//   description: string;
// }

const data = new Array(8).fill({
  title: 'Item',
  description: 'Description for Item',
});



// const SERVER_URL = 'http://192.168.1.10:5000';


export const HistoryScreen = ({ navigation }): React.ReactElement => {



  const controller = new AbortController();


  const [responseData, setResponseData] = React.useState<any>(null);



  // SERVER_URL

  // useFocusEffect

  useFocusEffect(
    React.useCallback(() => {
      // Update data from API
      fetch(SERVER_URL + "/loadHistory", {
        method: 'GET',
        // body: createFormData(response, { type }),
        // headers: {
        //   'Content-Type': 'multipart/form-data'
        // },
      })
        .then((response) => response.json())
        .then((responseJson) => {
          console.log("responseJson", responseJson);
          if (responseJson.success) {
            // setIsLoading(false)
            // setStep(1);
            setResponseData(responseJson.data);

          }

          // return responseJson.movies;
        })
        .catch((error) => {
          console.error(error);
          // setIsLoading(false)

        });


      return () => {
        // Clean up
        console.log('Screen lost focus');
      };
    }, [])
  );


  useEffect(() => {


  }, [])

  const renderItem = ({ item, index }: { item: any; index: number }): React.ReactElement => (
    <ListItem
      style={{
        backgroundColor: "#fce38a",
      }}
    // title={`${item.title} ${index + 1}`}
    // description={`${item.description} ${index + 1}`}
    >

      <View style={styles.itemContainer}>
        <Text style={styles.itemContainer1}>{moment(item.date * 1000).format('MM/DD/YYYY')}</Text>
        <Text style={styles.itemContainer2}>{get(item,"data.0.2","")}</Text>
        <Text style={styles.itemContainer3}>{get(item,"data.1.2","")}</Text>
      </View>


    </ListItem>
  );





  return (
    <KeyboardAvoidingView>
      <View style={styles.baseContainer}>

        <View>

          {/* <Text>{JSON.stringify(responseData)}</Text> */}

        </View>

        <View style={styles.itemContainer}>
          {/* {JSON.stringify(item)} */}
          <Text style={[styles.itemContainer1, { fontWeight: "900",fontSize:16 }]}>Date  </Text>
          <Text style={[styles.itemContainer2, { fontWeight: "900" }]}>Health</Text>
          <Text style={[styles.itemContainer3, { fontWeight: "900" }]}>Probability</Text>
        </View>

        {responseData && responseData && <List
          style={styles.container}
          data={responseData}

          ItemSeparatorComponent={Divider}
          renderItem={renderItem}
        />}



      </View>







    </KeyboardAvoidingView>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fce38a",
    // paddingVertical: 24,
    // paddingHorizontal: 16,
  },
  baseContainer: {
    // marginTop: 48,
    backgroundColor: "#fce38a",
    flex: 1
  },

  itemContainer: {
    flexDirection: "row",
    // paddingTop:20,
    marginTop: 10
    // marginTop: 48,
    // backgroundColor: "#fce38a",
    // flex: 1
  },

  itemContainer1: {
    width: "33%",
    textAlign: "center",
    color: "black"
    // marginTop: 48,
    // backgroundColor: "#fce38a",
    // flex: 1
  },

  itemContainer2: {
    width: "33%",
    textAlign: "center",
    color: "black",
    fontSize:16
    // marginTop: 48,
    // backgroundColor: "#fce38a",
    // flex: 1
  },

  itemContainer3: {
    width: "33%",
    textAlign: "center",
    color: "black"
    // marginTop: 48,
    // backgroundColor: "#fce38a",
    // flex: 1
  },


  signInContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 24,
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
