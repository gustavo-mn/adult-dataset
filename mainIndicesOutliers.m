clear all
close all
clc

load('../matriz-dissimilaridades-dataset-10pct.mat')

A = csvread('dataset-amostrado-10pct-sem-cabecalho.csv');

media_dissimilaridades = mean(D);

M = [media_dissimilaridades' A(:,1)];

M2 = sortrows(M,[1]);

figure; imagesc(D); colorbar
title('Matriz de Dissimilaridades', 'FontSize', 20)

figure;
plot(M2(:,1)); 
hold on; 
y = ylim; 
plot(ceil(0.9*size(M2,1))*ones(100,1), linspace(y(1),y(2),100), 'r'); 
hold off;

title('Dissimilaridades médias por registro (ordem crescente)', 'FontSize', 20)

indices_registros_outliers = M2(end-ceil(0.1*size(M2,1))+1:end, 2);

%csvwrite('indices-outliers-dataset-10pct.csv', indices_registros_outliers)