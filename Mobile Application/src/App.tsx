/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React from 'react';
import type { PropsWithChildren } from 'react';
import { ApplicationProvider, IconRegistry } from '@ui-kitten/components';
// import { AppStorage } from './services/app-storage.service';
// import { Mapping, Theme, Theming } from './services/theme.service';
// import { AppLoading, LoadFontsTask, Task } from './app/app-loading.component';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { EvaIconsPack } from '@ui-kitten/eva-icons';
import { DefaultTheme, NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { mapping, dark as drakTheme } from '@eva-design/eva';


import {
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  useColorScheme,
  View,
} from 'react-native';



import {
  BottomTabNavigationOptions,
  createBottomTabNavigator,
} from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { HomeBottomNavigation } from './scenes/home/home-bottom-navigation.component';
import { LayoutsScreen } from './scenes/layouts/layouts.component';
import { DashboardsNavigator } from './navigation/dashboards.navigator';
import { createMaterialTopTabNavigator } from '@react-navigation/material-top-tabs';
import { TabBar, Tab, Layout, Text } from '@ui-kitten/components';
import { HomeScreen } from './scenes/HomeScreen';
import { HealthyScreen } from './scenes/HealthyScreen';

import { HistoryScreen } from './scenes/HistoryScreen';



export const SERVER_URL = 'http://192.168.1.10:5000';

// export const SERVER_URL = 'http://192.168.86.24:5000';



/*
 * Navigation theming: https://reactnavigation.org/docs/en/next/themes.html
 */
const navigatorTheme = {
  // ...DefaultTheme,
  colors: {
    // ...DefaultTheme.colors,
    ...{
      primary: 'rgb(0, 0, 255)',
      background: 'rgb(242, 242, 242)',
      card: 'rgb(255, 255, 255)',
      text: 'rgb(28, 28, 30)',
      border: 'rgb(216, 216, 216)',
      notification: 'rgb(255, 59, 48)',
    },
    // prevent layout blinking when performing navigation
    background: 'transparent',
  },
};

const ROOT_ROUTES: string[] = ['Home', 'Layouts', 'Components', 'Themes'];

const TabBarVisibilityOptions = ({ route }): BottomTabNavigationOptions => {
  const isNestedRoute: boolean = route.state?.index > 0;
  const isRootRoute: boolean = ROOT_ROUTES.includes(route.name);

  return { tabBarVisible: isRootRoute && !isNestedRoute };
};
const initialTabRoute: string = __DEV__ ? 'Components' : 'Layouts';

const BottomTab = createBottomTabNavigator();
const Drawer = createDrawerNavigator();
const Stack = createStackNavigator();

export const LayoutsNavigator = (): React.ReactElement => (
  <Stack.Navigator headerMode='none'>
    <Stack.Screen name='Home' component={LayoutsScreen} />
    <Stack.Screen name='Dashboards' component={DashboardsNavigator} />
    {/* <Stack.Screen name='Auth' component={AuthNavigator}/>
    <Stack.Screen name='Social' component={SocialNavigator}/>
    <Stack.Screen name='Articles' component={ArticlesNavigator}/>
    <Stack.Screen name='Messaging' component={MessagingNavigator}/>
    <Stack.Screen name='Dashboards' component={DashboardsNavigator}/>
    <Stack.Screen name='Ecommerce' component={EcommerceNavigator}/> */}
  </Stack.Navigator>
);

const HomeTabsNavigator = (): React.ReactElement => (
  <BottomTab.Navigator
    screenOptions={TabBarVisibilityOptions}
    initialRouteName={initialTabRoute}
    tabBar={props => <HomeBottomNavigation {...props} />}>
    <BottomTab.Screen name='Layouts' component={LayoutsNavigator} />
    <BottomTab.Screen name='Components' component={LayoutsNavigator} />
    {/* <BottomTab.Screen name='Themes' component={ThemesNavigator} /> */}
  </BottomTab.Navigator>
);

const { Navigator, Screen } = createMaterialTopTabNavigator();

// const HomeScreen = () => (
//   <Layout style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
//     <Text category='h1'>USERS</Text>

//   </Layout>
// );

// const HealthyScreen = () => (
//   <Layout style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
//     <Text category='h1'>ORDERS</Text>
//   </Layout>
// );

// const HistoryScreen = () => (
//   <Layout style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
//     <Text category='h1'>History</Text>
//   </Layout>
// );

const TopTabBar = ({ navigation, state }) => (
  <TabBar

    selectedIndex={state.index}
    onSelect={index => navigation.navigate(state.routeNames[index])}>
    <Tab title='Weather' />
    <Tab title='Health' />
    <Tab title='History' />
  </TabBar>
);

const TabNavigator = () => (
  <Navigator

    // screenOptions={{
    //   tabBarActiveTintColor: '#e91e63',
    //   tabBarLabelStyle: { fontSize: 20 },
    //   tabBarStyle: { backgroundColor: 'powderblue' },
    // }}

    tabBar={props => <TopTabBar {...props} />}



  >

    {/* <HomeScreen */}
    <Screen name='Home' component={HomeScreen} />
    <Screen name='Healthy' component={HealthyScreen} />
    <Screen name='History' component={HistoryScreen} />
  </Navigator>
);

export const AppNavigator = (): React.ReactElement => (
  <NavigationContainer  >
    <TabNavigator />

    {/* WebViewScreen */}

    {/* <WebViewScreen></WebViewScreen> */}

  </NavigationContainer>
);

const App: React.FC = () => {

  // const [mappingContext, currentMapping] = Theming.useMapping(appMappings, mapping);
  // const [themeContext, currentTheme] = Theming.useTheming(appThemes, mapping, theme);

  return (
    <React.Fragment>
      <IconRegistry icons={[EvaIconsPack]} />

      <ApplicationProvider
        mapping={mapping}
        theme={drakTheme}
      >

        <SafeAreaProvider>
          <StatusBar />
          <AppNavigator></AppNavigator>
        </SafeAreaProvider>
      </ApplicationProvider>

    </React.Fragment>
  );
};

// function App(): React.JSX.Element {
//   const isDarkMode = useColorScheme() === 'dark';

//   const backgroundStyle = {
//     backgroundColor: isDarkMode ? Colors.darker : Colors.lighter,
//   };

//   return (
//     <SafeAreaView style={backgroundStyle}>
//       <StatusBar
//         barStyle={isDarkMode ? 'light-content' : 'dark-content'}
//         backgroundColor={backgroundStyle.backgroundColor}
//       />
//       <ScrollView
//         contentInsetAdjustmentBehavior="automatic"
//         style={backgroundStyle}>
//         <Header />
//         <View
//           style={{
//             backgroundColor: isDarkMode ? Colors.black : Colors.white,
//           }}>
//           <Section title="Step One">
//             Edit <Text style={styles.highlight}>App.tsx</Text> to change this
//             screen and then come back to see your edits.
//           </Section>
//           <Section title="See Your Changes">
//             <ReloadInstructions />
//           </Section>
//           <Section title="Debug">
//             <DebugInstructions />
//           </Section>
//           <Section title="Learn More">
//             Read the docs to discover what to do next:
//           </Section>
//           <LearnMoreLinks />
//         </View>
//       </ScrollView>
//     </SafeAreaView>
//   );
// }

const styles = StyleSheet.create({
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  sectionTitle: {
    fontSize: 24,
    fontWeight: '600',
  },
  sectionDescription: {
    marginTop: 8,
    fontSize: 18,
    fontWeight: '400',
  },
  highlight: {
    fontWeight: '700',
  },
});

export default App;
