(define (problem tireworld-083-03)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l9 - location)
    (:init (road l0 l4) (road l0 l6) (road l1 l35) (road l1 l76) (road l10 l2) (road l10 l47) (road l11 l19) (road l11 l81) (road l12 l48) (road l12 l56) (road l13 l53) (road l13 l77) (road l14 l21) (road l14 l42) (road l14 l60) (road l15 l32) (road l15 l49) (road l16 l25) (road l16 l69) (road l17 l2) (road l17 l60) (road l18 l32) (road l18 l62) (road l19 l11) (road l19 l29) (road l2 l10) (road l2 l17) (road l20 l22) (road l20 l25) (road l21 l14) (road l21 l48) (road l22 l20) (road l22 l24) (road l23 l35) (road l23 l72) (road l24 l22) (road l24 l67) (road l25 l16) (road l25 l20) (road l26 l34) (road l26 l40) (road l26 l61) (road l26 l65) (road l27 l6) (road l27 l73) (road l28 l70) (road l28 l77) (road l29 l19) (road l29 l4) (road l3 l42) (road l3 l54) (road l30 l31) (road l30 l8) (road l31 l30) (road l31 l71) (road l32 l15) (road l32 l18) (road l33 l58) (road l33 l66) (road l34 l26) (road l34 l52) (road l35 l1) (road l35 l23) (road l36 l50) (road l37 l52) (road l37 l53) (road l38 l7) (road l38 l75) (road l39 l82) (road l39 l9) (road l4 l0) (road l4 l29) (road l40 l26) (road l40 l54) (road l40 l63) (road l41 l68) (road l41 l81) (road l42 l14) (road l42 l3) (road l43 l51) (road l43 l71) (road l44 l79) (road l44 l8) (road l45 l74) (road l45 l82) (road l46 l57) (road l46 l9) (road l47 l10) (road l47 l5) (road l47 l63) (road l48 l12) (road l48 l21) (road l48 l72) (road l49 l15) (road l49 l50) (road l5 l47) (road l5 l66) (road l50 l36) (road l50 l49) (road l51 l43) (road l51 l69) (road l52 l34) (road l52 l37) (road l52 l76) (road l53 l13) (road l53 l37) (road l54 l3) (road l54 l40) (road l54 l61) (road l55 l74) (road l55 l78) (road l56 l12) (road l56 l67) (road l57 l46) (road l57 l73) (road l58 l33) (road l58 l80) (road l59 l64) (road l59 l75) (road l6 l0) (road l6 l27) (road l60 l14) (road l60 l17) (road l61 l26) (road l61 l54) (road l61 l70) (road l61 l72) (road l62 l18) (road l62 l64) (road l63 l40) (road l63 l47) (road l64 l59) (road l64 l62) (road l65 l26) (road l65 l79) (road l66 l33) (road l66 l5) (road l67 l24) (road l67 l56) (road l68 l41) (road l68 l80) (road l69 l16) (road l69 l51) (road l7 l38) (road l7 l8) (road l70 l28) (road l70 l61) (road l71 l31) (road l71 l43) (road l72 l23) (road l72 l48) (road l72 l61) (road l73 l27) (road l73 l57) (road l74 l45) (road l74 l55) (road l75 l38) (road l75 l59) (road l76 l1) (road l76 l52) (road l77 l13) (road l77 l28) (road l78 l55) (road l79 l44) (road l79 l65) (road l8 l30) (road l8 l44) (road l8 l7) (road l80 l58) (road l80 l68) (road l81 l11) (road l81 l41) (road l82 l39) (road l82 l45) (road l9 l39) (road l9 l46) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l11) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l19) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l31) (spare-in l32) (spare-in l33) (spare-in l35) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l43) (spare-in l45) (spare-in l46) (spare-in l47) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l54) (spare-in l55) (spare-in l56) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l60) (spare-in l61) (spare-in l62) (spare-in l64) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l7) (spare-in l70) (spare-in l71) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l79) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l9) (vehicle-at l36))
    (:goal (vehicle-at l78))
)