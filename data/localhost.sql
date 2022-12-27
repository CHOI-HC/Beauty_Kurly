-- phpMyAdmin SQL Dump
-- version 3.2.3
-- http://www.phpmyadmin.net
--
-- 호스트: localhost
-- 처리한 시간: 22-12-07 11:33 
-- 서버 버전: 5.1.41
-- PHP 버전: 5.2.12

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- 데이터베이스: `kurlydb`
--
CREATE DATABASE `kurlydb` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE `kurlydb`;

-- --------------------------------------------------------

--
-- 테이블 구조 `beauty`
--

CREATE TABLE IF NOT EXISTS `beauty` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `cate` varchar(10) NOT NULL,
  `title` varchar(120) NOT NULL DEFAULT '',
  `coupon` int(11) NOT NULL,
  `price` int(11) NOT NULL,
  `oriprice` int(11) NOT NULL,
  `pcode` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `pcode` (`pcode`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

--
-- 테이블의 덤프 데이터 `beauty`
--


-- --------------------------------------------------------

--
-- 테이블 구조 `breview`
--

CREATE TABLE IF NOT EXISTS `breview` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `pcode` int(11) NOT NULL,
  `rkey` varchar(120) NOT NULL DEFAULT '',
  `review` varchar(500) NOT NULL DEFAULT '',
  `date` date NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `rkey` (`rkey`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1 ;

--
-- 테이블의 덤프 데이터 `breview`
--

