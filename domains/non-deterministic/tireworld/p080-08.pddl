(define (problem tireworld-080-08)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l9 - location)
    (:init (road l0 l61) (road l0 l69) (road l1 l47) (road l1 l68) (road l10 l56) (road l10 l8) (road l11 l23) (road l11 l62) (road l12 l29) (road l12 l45) (road l12 l66) (road l13 l22) (road l13 l36) (road l14 l43) (road l14 l78) (road l15 l57) (road l15 l60) (road l16 l4) (road l16 l57) (road l17 l71) (road l18 l32) (road l18 l59) (road l19 l63) (road l19 l74) (road l2 l40) (road l2 l74) (road l20 l30) (road l20 l33) (road l21 l64) (road l21 l79) (road l22 l13) (road l22 l6) (road l23 l11) (road l23 l65) (road l24 l37) (road l24 l70) (road l25 l31) (road l25 l47) (road l26 l46) (road l26 l71) (road l27 l40) (road l27 l75) (road l28 l43) (road l28 l60) (road l29 l12) (road l29 l42) (road l3 l50) (road l3 l54) (road l30 l20) (road l30 l50) (road l31 l25) (road l31 l44) (road l32 l18) (road l32 l33) (road l33 l20) (road l33 l32) (road l34 l64) (road l34 l66) (road l35 l38) (road l35 l51) (road l36 l13) (road l36 l46) (road l37 l24) (road l37 l49) (road l38 l35) (road l38 l48) (road l39 l67) (road l39 l76) (road l4 l16) (road l4 l59) (road l40 l2) (road l40 l27) (road l41 l53) (road l41 l76) (road l42 l29) (road l42 l9) (road l43 l14) (road l43 l28) (road l44 l31) (road l44 l52) (road l45 l12) (road l45 l78) (road l46 l26) (road l46 l36) (road l46 l56) (road l47 l1) (road l47 l25) (road l48 l38) (road l48 l58) (road l49 l37) (road l49 l75) (road l5 l55) (road l5 l7) (road l50 l3) (road l50 l30) (road l51 l35) (road l51 l73) (road l52 l44) (road l52 l72) (road l53 l41) (road l53 l77) (road l54 l3) (road l54 l73) (road l55 l5) (road l55 l72) (road l56 l10) (road l56 l46) (road l57 l15) (road l57 l16) (road l58 l48) (road l59 l18) (road l59 l4) (road l6 l22) (road l6 l65) (road l60 l15) (road l60 l28) (road l61 l0) (road l61 l70) (road l62 l11) (road l62 l79) (road l63 l19) (road l63 l8) (road l64 l21) (road l64 l34) (road l65 l23) (road l65 l6) (road l66 l12) (road l66 l34) (road l67 l39) (road l67 l69) (road l68 l1) (road l68 l77) (road l69 l0) (road l69 l67) (road l7 l5) (road l7 l9) (road l70 l24) (road l70 l61) (road l71 l17) (road l71 l26) (road l72 l52) (road l72 l55) (road l73 l51) (road l73 l54) (road l74 l19) (road l74 l2) (road l75 l27) (road l75 l49) (road l76 l39) (road l76 l41) (road l77 l53) (road l77 l68) (road l78 l14) (road l78 l45) (road l79 l21) (road l79 l62) (road l8 l10) (road l8 l63) (road l9 l42) (road l9 l7) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l37) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l57) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l79) (spare-in l8) (spare-in l9) (vehicle-at l17))
    (:goal (vehicle-at l58))
)