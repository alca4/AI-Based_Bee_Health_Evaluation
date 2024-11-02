import Constants from 'expo-constants';

import * as RNComponent from './app-loading.component.rn';

export type TaskResult<T = any> = [string, T];
export type Task = () => Promise<TaskResult | null>;

const Component =  RNComponent;

export const AppLoading = RNComponent.AppLoading;
export const LoadFontsTask = RNComponent.LoadFontsTask;
